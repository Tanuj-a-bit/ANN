import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwritingModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super(HandwritingModel, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2) # 32x128 -> 16x64
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2) # 16x64 -> 8x32
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1)) # 8x32 -> 4x32
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1)) # 4x32 -> 2x32
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        # Final feature map size: 1 x 32 x 512 (approx)
        
        # RNN Layers (Bidirectional LSTM)
        self.lstm = nn.LSTM(1024, hidden_size, num_layers, 
                            bidirectional=True, batch_first=True, dropout=0.2)
        
        # Output fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 32, 128)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Current shape: (batch_size, 512, 2, 32)
        # Reshape for LSTM: (batch_size, sequence_length, features)
        # We want to treat the horizontal dimension as the sequence
        x = x.permute(0, 3, 1, 2) # (batch_size, 32, 512, 2)
        batch_size, seq_len, channels, height = x.size()
        x = x.reshape(batch_size, seq_len, channels * height)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Prediction
        x = self.fc(x)
        
        # Return log probabilities for CTC loss
        # Expected shape by CTC: (sequence_length, batch_size, num_classes)
        return x.permute(1, 0, 2).log_softmax(2)

if __name__ == "__main__":
    from config import NUM_CLASSES
    model = HandwritingModel(num_classes=NUM_CLASSES)
    test_input = torch.randn(1, 1, 32, 128)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}") # Should be (32, 1, NUM_CLASSES)
