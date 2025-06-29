# Heatwave Forecasting Project - Workflow Diagram

## 🔄 Complete Project Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HEATWAVE FORECASTING PROJECT                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INPUT DATA    │    │   INPUT DATA    │    │   INPUT DATA    │
│                 │    │                 │    │                 │
│ • NetCDF Files  │    │ • NetCDF Files  │    │ • NetCDF Files  │
│ • d2m variable  │    │ • d2m variable  │    │ • d2m variable  │
│ • t2m variable  │    │ • t2m variable  │    │ • t2m variable  │
│ • 8832 timesteps│    │ • 8832 timesteps│    │ • 8832 timesteps│
│ • 101x161 grid  │    │ • 101x161 grid  │    │ • 101x161 grid  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   DATA PREPROCESSING    │
                    │                         │
                    │ 1. Load NetCDF files    │
                    │ 2. Spatial downsampling │
                    │ 3. Temporal sampling    │
                    │ 4. Normalization        │
                    │ 5. Sequence creation    │
                    │ 6. Train/Val/Test split │
                    │ 7. Save compressed data │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   PROCESSED DATA        │
                    │                         │
                    │ • X_train.npz (16,685)  │
                    │ • y_train.npz (16,685)  │
                    │ • X_val.npz (3,575)     │
                    │ • y_val.npz (3,575)     │
                    │ • X_test.npz (3,576)    │
                    │ • y_test.npz (3,576)    │
                    │ • scalers.pkl           │
                    │ • metadata.pkl          │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   MODEL TRAINING        │
                    │                         │
                    │ ┌─────────┐ ┌─────────┐ │
                    │ │   CNN   │ │CNN-LSTM │ │
                    │ │ Model   │ │ Model   │ │
                    │ └─────┬───┘ └─────┬───┘ │
                    │       │           │     │
                    │ ┌─────▼───┐       │     │
                    │ │CNN-LSTM │       │     │
                    │ │ -GRU    │       │     │
                    │ │ Model   │       │     │
                    │ └─────────┘       │     │
                    └─────────┬─────────┼─────┘
                              │         │
                              ▼         ▼
                    ┌─────────────────────────┐
                    │   TRAINED MODELS        │
                    │                         │
                    │ • cnn_best_loss.pth     │
                    │ • cnn_best_r2.pth       │
                    │ • cnn_lstm_best_loss.pth│
                    │ • cnn_lstm_best_r2.pth  │
                    │ • cnn_lstm_gru_best_loss│
                    │ • cnn_lstm_gru_best_r2  │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   MODEL EVALUATION      │
                    │                         │
                    │ 1. Load test data       │
                    │ 2. Generate predictions │
                    │ 3. Calculate metrics    │
                    │ 4. Event detection      │
                    │ 5. Threshold optimization│
                    │ 6. Generate visualizations│
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   RESULTS & OUTPUTS     │
                    │                         │
                    │ 📊 Performance Metrics: │
                    │ • R² scores             │
                    │ • RMSE values           │
                    │ • MAE values            │
                    │                         │
                    │ 🌡️ Event Detection:     │
                    │ • Heatwave F1 scores    │
                    │ • Coldwave F1 scores    │
                    │ • Precision/Recall      │
                    │                         │
                    │ 📈 Visualizations:      │
                    │ • Model comparison plots│
                    │ • Time series analysis  │
                    │ • Spatial event maps    │
                    │ • Seasonal analysis     │
                    └─────────────────────────┘
```

## 🔧 Detailed Process Flow

### 1. Data Preprocessing Pipeline
```
NetCDF Files
    ↓
Load & Concatenate
    ↓
Spatial Downsampling (2x reduction)
    ↓
Temporal Sampling (90% of timesteps)
    ↓
Normalization (StandardScaler)
    ↓
Sequence Creation (10 timesteps → 1 prediction)
    ↓
Train/Validation/Test Split (70/15/15)
    ↓
Compressed Storage (.npz format)
```

### 2. Model Training Pipeline
```
Processed Data
    ↓
DataLoader Creation
    ↓
Model Initialization
    ↓
Training Loop (20 epochs)
    ├── Forward Pass
    ├── Loss Calculation (MSE)
    ├── Backward Pass
    ├── Parameter Update (Adam)
    └── Validation Check
    ↓
Early Stopping (patience=10)
    ↓
Model Saving (best loss & R²)
```

### 3. Evaluation Pipeline
```
Test Data + Trained Models
    ↓
Batch Prediction Generation
    ↓
Metric Calculation
    ├── R² Score
    ├── RMSE
    ├── MAE
    └── Bias
    ↓
Event Detection
    ├── Threshold Optimization
    ├── Heatwave Detection
    └── Coldwave Detection
    ↓
Visualization Generation
    ├── Performance Comparison
    ├── Time Series Analysis
    ├── Spatial Maps
    └── Seasonal Analysis
```

## 📊 Data Dimensions Flow

### Input Data
- **Raw**: (8832, 101, 161, 2) - 8.8GB
- **Downsampled**: (8832, 51, 81, 2) - 2.2GB
- **Temporal Sampled**: (7949, 51, 81, 2) - 2.0GB
- **Sequenced**: (7939, 10, 51, 81, 2) - 20.0GB
- **Final Split**: 
  - Train: (16,685, 10, 51, 81, 2) - 5.1GB
  - Val: (3,575, 10, 51, 81, 2) - 1.1GB
  - Test: (3,576, 10, 51, 81, 2) - 1.1GB

### Model Input/Output
- **CNN Input**: (batch, 1, 51, 81, 2)
- **LSTM Input**: (batch, 10, 51, 81, 2)
- **Output**: (batch, 51, 81, 2)

## ⚡ Performance Optimization

### Memory Management
- **Float32 precision**: 50% memory reduction
- **Compressed storage**: .npz format
- **Batch processing**: 16 samples per batch
- **GPU acceleration**: CUDA support

### Training Optimization
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Adaptive optimization
- **Batch normalization**: Stable training
- **Dropout**: Regularization

## 🎯 Key Decision Points

1. **Spatial Resolution**: 2x downsampling for memory vs. accuracy
2. **Temporal Sampling**: 90% retention for pattern preservation
3. **Model Selection**: CNN vs. CNN-LSTM vs. CNN-LSTM-GRU
4. **Threshold Optimization**: F1-score based event detection
5. **Evaluation Metrics**: R², RMSE, MAE, F1-score

## 📈 Success Metrics

- **Temperature R²**: > 0.6 (achieved: 0.6782)
- **Dewpoint R²**: > 0.5 (achieved: 0.6191)
- **Heatwave F1**: > 0.8 (achieved: 0.8438)
- **Coldwave F1**: > 0.8 (achieved: 0.8438)
- **Training Time**: < 1 hour per model
- **Memory Usage**: < 8GB total dataset 