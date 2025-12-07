"""
CNN model architecture for mel-spectrogram classification.
"""

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    Convolutional Neural Network for music genre classification.
    Processes mel-spectrograms.
    """
    
    def __init__(self, num_classes, conv_channels=[32, 64], kernel_size=3, 
                 pool_size=2, dense_units=[128, 64], dropout=0.4):
        """
        Initialize CNN model.
        
        Args:
            num_classes: Number of output classes
            conv_channels: List of channel sizes for conv layers
            kernel_size: Kernel size for convolutions
            pool_size: Pool size for max pooling
            dense_units: List of units for dense layers
            dropout: Dropout rate
        """
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, conv_channels[0], kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.pool1 = nn.MaxPool2d(pool_size)
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], 
                              kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.pool2 = nn.MaxPool2d(pool_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.flatten = nn.Flatten()
        
        self.fc_layers = nn.ModuleList()
        
        self._feature_size = None
        
        prev_units = None
        for units in dense_units:
            if prev_units is None:
                prev_units = units
                continue
            self.fc_layers.append(nn.Linear(prev_units, units))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            prev_units = units
        
        self.dense_units = dense_units
        self.output = None
    
    def _get_conv_output_size(self, input_shape):
        """Calculate the output size after conv layers."""
        batch_size = 1
        input = torch.zeros(batch_size, *input_shape)
        
        x = self.pool1(self.bn1(torch.relu(self.conv1(input))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.flatten(x)
        
        return x.size(1)
    
    def build_fc_layers(self, input_shape, num_classes):
        """Build fully connected layers after knowing conv output size."""
        self._feature_size = self._get_conv_output_size(input_shape)
        
        self.fc_layers = nn.ModuleList()
        prev_units = self._feature_size
        
        for units in self.dense_units:
            self.fc_layers.append(nn.Linear(prev_units, units))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(self.dropout.p))
            prev_units = units
        
        self.output = nn.Linear(prev_units, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        
        for layer in self.fc_layers:
            x = layer(x)
        
        if self.output is not None:
            x = self.output(x)
        
        return x
