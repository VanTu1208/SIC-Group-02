import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Data augmentation and preprocessing
train_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

training_dir = pathlib.Path("./training_set")

# Load dataset with appropriate transforms
full_dataset = datasets.ImageFolder(root=str(training_dir), transform=None)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Use fixed random generator for reproducible splits
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=generator
)


# Apply transforms after splitting
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# MixUp data augmentation
class MixUpDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]

        # Sample another image/label randomly
        if self.alpha > 0:
            idx2 = random.randint(0, len(self.dataset) - 1)
            img2, label2 = self.dataset[idx2]

            # Generate mixup coefficient
            lam = np.random.beta(self.alpha, self.alpha)

            # Mix images and labels
            img = lam * img1 + (1 - lam) * img2

            # One-hot encode labels
            label1_one_hot = F.one_hot(
                torch.tensor(label1), num_classes=len(full_dataset.classes)
            ).float()
            label2_one_hot = F.one_hot(
                torch.tensor(label2), num_classes=len(full_dataset.classes)
            ).float()

            # Mix labels
            mixed_label = lam * label1_one_hot + (1 - lam) * label2_one_hot

            return img, mixed_label
        else:
            # If alpha <= 0, return original sample
            return (
                img1,
                F.one_hot(
                    torch.tensor(label1), num_classes=len(full_dataset.classes)
                ).float(),
            )


# Apply transforms and mixup
train_dataset = TransformedSubset(train_dataset, train_transform)
train_dataset = MixUpDataset(train_dataset, alpha=0.2)
val_dataset = TransformedSubset(val_dataset, val_transform)

class_names = full_dataset.classes
num_classes = len(class_names)

# Dataloaders with appropriate batch sizes
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Classes: {class_names}")
print(f"Training samples: {train_size}, Validation samples: {val_size}")


# Squeeze and Excitation block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Depthwise Separable Convolution for parameter efficiency
class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Dual Path Block - combines ResNet and DenseNet ideas
class DualPathBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DualPathBlock, self).__init__()

        # Bottleneck design
        bottleneck_channels = out_channels // 4

        # First convolution - reduce dimensions
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # Depthwise separable convolution - efficient spatial processing
        self.conv2 = DepthwiseSeparableConv(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # Expand dimensions
        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Squeeze and Excitation for channel attention
        self.se = SEBlock(out_channels, reduction=16)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply SE attention
        out = self.se(out)

        # Add residual connection
        out += self.shortcut(residual)
        out = self.relu(out)

        return out


# Spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and sigmoid
        x_out = self.conv(x_cat)
        attention = self.sigmoid(x_out)

        # Apply attention
        return x * attention


# Efficient Dual-Path Network
class EfficientDualPathNet(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientDualPathNet, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Dual Path Blocks with increasing channels
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 96, blocks=2, stride=2)
        self.layer3 = self._make_layer(96, 128, blocks=1, stride=2)

        # Spatial attention after feature extraction
        self.spatial_attention = SpatialAttention(kernel_size=5)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []

        # First block with stride
        layers.append(DualPathBlock(in_channels, out_channels, stride=stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(DualPathBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Dual path blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Apply spatial attention
        x = self.spatial_attention(x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x  # No softmax for mixup training


# Create model and print parameter count
model = EfficientDualPathNet(num_classes=num_classes).to(device)


# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params:,}")

# Ensure we're under the 300,000 parameter limit
assert (
    total_params < 300000
), f"Model has {total_params} parameters, exceeding the 300,000 limit"

# Loss function for mixup training
criterion = nn.BCEWithLogitsLoss()

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Cosine annealing learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)


# Training function with early stopping
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=30,
    patience=7,
):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # For mixup training
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            # For accuracy calculation with mixup
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct += (predicted == labels_max).sum().item()
            total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix(
                loss=running_loss / len(train_bar), acc=correct / total
            )

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_running_loss, correct, total = 0.0, 0, 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # For validation, we use standard cross-entropy
                loss = F.cross_entropy(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                val_bar.set_postfix(
                    loss=val_running_loss / len(val_bar), acc=correct / total
                )

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        # Update learning rate with cosine annealing
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch results
        print(
            f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss, val_loss, train_acc, val_acc


# Train the model with more epochs and early stopping
epochs = 30
model, train_loss, val_loss, train_acc, val_acc = train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=epochs,
    patience=7,
)

# Save the model
example_input = torch.randn(1, 3, 128, 128).to(device)
traced_model = torch.jit.trace(model, example_input)

# sinh vien luu model voi name la mssv
traced_model.save("22119226.pt")
print("Model saved as 22119226.pt")

# Print final model performance
print(f"Final training accuracy: {train_acc[-1]:.4f}")
print(f"Final validation accuracy: {val_acc[-1]:.4f}")
