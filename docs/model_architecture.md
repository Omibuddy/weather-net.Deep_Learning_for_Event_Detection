# Model Architecture Details

## Overview
The model architecture combines CNN, LSTM, and attention mechanisms for heatwave forecasting. Here's a detailed breakdown:

## Input Layer
- Shape: (batch_size, seq_len=7, num_features=13)
- Features: Temperature, Humidity, Pressure, etc.

## Feature Extraction
```python
self.feature_extractor = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU()
)
```

## CNN Component
```python
# Multi-scale CNN
self.conv1d_3 = nn.Conv1d(num_features, 128, kernel_size=3, padding=1)
self.conv1d_5 = nn.Conv1d(num_features, 128, kernel_size=5, padding=2)
self.conv1d_7 = nn.Conv1d(num_features, 64, kernel_size=7, padding=3)

# Residual blocks
self.res_blocks = nn.ModuleList([
    ResidualBlock(320),  # 128 + 128 + 64
    ResidualBlock(320),
    ResidualBlock(320)
])
```

## LSTM Component
```python
self.lstm1 = nn.LSTM(
    input_size=256, 
    hidden_size=lstm_units,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=True
)

self.lstm2 = nn.LSTM(
    input_size=2 * lstm_units,
    hidden_size=lstm_units//2,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)
```

## Attention Mechanism
```python
self.attention = nn.MultiheadAttention(
    embed_dim=lstm_units,
    num_heads=16,
    dropout=0.3,
    batch_first=True
)
```

## Output Layer
```python
self.dense1 = nn.Linear(combined_features, dense_units)
self.dense2 = nn.Linear(dense_units, dense_units // 2)
self.dense3 = nn.Linear(dense_units // 2, dense_units // 4)
self.output = nn.Linear(dense_units // 4, num_classes)
```

## Model Parameters
- Total parameters: ~2.5M
- Trainable parameters: ~2.5M
- Non-trainable parameters: 0

## Layer Dimensions
1. Input: (batch_size, 7, 13)
2. Feature Extraction: (batch_size, 7, 32)
3. CNN Output: (batch_size, 7, 256)
4. LSTM1 Output: (batch_size, 7, 256)
5. LSTM2 Output: (batch_size, 7, 128)
6. Attention Output: (batch_size, 7, 128)
7. Final Output: (batch_size, 1)

## Training Configuration
- Batch size: 32
- Learning rate: 0.001
- Optimizer: AdamW
- Weight decay: 0.01
- Gradient clipping: 1.0
- Early stopping patience: 15
- Maximum epochs: 100

## Regularization Techniques
1. Dropout
   - Feature extraction: 0.2
   - LSTM: 0.3
   - Attention: 0.3
   - Dense layers: 0.2

2. Batch Normalization
   - After CNN layers
   - Before LSTM layers

3. Layer Normalization
   - After LSTM layers
   - Before attention mechanism

## Data Augmentation
1. Gaussian Noise
   - Mean: 0
   - Standard deviation: 0.01

2. Time Warping
   - Sigma: 0.2
   - Range: [0.8, 1.2]

3. Feature Masking
   - Probability: 0.1
   - Mask value: 0 