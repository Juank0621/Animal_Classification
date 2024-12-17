import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import Swin_T_Weights, ConvNeXt_Base_Weights

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Block 1: Convolution, Batch Normalization, Max Pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Convolution, Batch Normalization, Max Pooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: Convolution, Batch Normalization, Max Pooling
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: Convolution, Batch Normalization, Max Pooling
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5: Convolution, Batch Normalization, Max Pooling
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 6: Convolution, Batch Normalization, Max Pooling
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the input size for the fully connected layer
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024 * 3 * 3, 512)  # Adjust the input size based on the output of the last pooling layer
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.silu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        # Block 2
        x = F.silu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        # Block 3
        x = F.silu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)

        # Block 4
        x = F.silu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)

        # Block 5
        x = F.silu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool5(x)

        # Block 6
        x = F.silu(self.conv6(x))
        x = self.bn6(x)
        x = self.pool6(x)

        # Classifier
        x = self.flatten(x)
        x = self.dropout1(x)
        x = F.silu(self.fc1(x))
        x = self.bn_fc1(x)
        x = self.dropout2(x)
        x = F.silu(self.fc2(x))
        x = self.bn_fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)

        return x

class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()

        # Load the EfficientNet-B4 model pre-trained on ImageNet
        self.base_model = models.efficientnet_b4(weights=None)  # Use weights=None
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])  # Remove the last two layers (avgpool and classifier)

        # Unfreeze EfficientNet-B4 layers to allow retraining
        for param in self.base_model.parameters():
            param.requires_grad = True

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1792, 1024),  # EfficientNet-B4 has 1792 features in the last conv layer
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SwinTransformerTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerTransferLearning, self).__init__()

        # Load the pre-trained Swin Transformer model on ImageNet
        self.base_model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

        # Freeze the base model parameters to avoid retraining them
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Replace the classifier head for transfer learning
        # Get the input features of the original head (the final classifier layer)
        in_features = self.base_model.head.in_features
        # Set a new fully connected layer with the number of desired output classes
        self.base_model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Forward pass through the base model
        x = self.base_model(x)
        return x

class ConvNeXtTransferLearning(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXtTransferLearning, self).__init__()

        # Load the ConvNeXt model pre-trained on ImageNet
        self.base_model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)  # Load pre-trained weights

        # Freeze the base model parameters to avoid retraining them
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Replace the classifier head for transfer learning
        in_features = self.base_model.classifier[2].in_features  # Get the input features of the original head
        self.base_model.classifier[2] = nn.Linear(in_features, num_classes)  # New head with desired number of classes

    def forward(self, x):
        x = self.base_model(x)
        return x



