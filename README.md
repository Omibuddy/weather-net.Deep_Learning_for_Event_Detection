# Heatwave Forecasting Project

##  Project Overview

This project implements deep learning models for forecasting heatwave and coldwave events using climate data. The system processes NetCDF climate data files containing temperature and dewpoint measurements to predict extreme weather events with high accuracy.

##  Data Description

### Input Data
- **Format**: NetCDF (.nc) files
- **Variables**: 
  - `d2m`: Dewpoint temperature at 2 meters
  - `t2m`: Air temperature at 2 meters
- **Dimensions**: 
  - Time: 8,832 timesteps
  - Latitude: 101 points
  - Longitude: 161 points
- **Data Quality**: No missing values, high-quality climate data

### Data Processing
The preprocessing pipeline includes:
- **Spatial downsampling**: Reduces resolution while preserving patterns
- **Temporal sampling**: Intelligent sampling to reduce memory usage
- **Normalization**: StandardScaler for each variable
- **Sequence creation**: 10-timestep input sequences for 1-timestep predictions
- **Memory optimization**: Float32 precision and compressed storage

##  Model Architecture

### 1. CNN Model (`cnn_model.py`)
- **Architecture**: Convolutional Neural Network
- **Features**: 
  - 3 convolutional blocks with batch normalization
  - Max pooling for spatial reduction
  - Dense regression head
- **Input**: Last timestep of sequence (1, height, width, channels)
- **Output**: Predicted temperature and dewpoint maps

### 2. CNN-LSTM Model (`cnn_lstm_model.py`)
- **Architecture**: CNN + LSTM hybrid
- **Features**:
  - CNN for spatial feature extraction
  - LSTM layers for temporal dependencies
  - Attention mechanism
- **Input**: Full sequence (10, height, width, channels)
- **Output**: Predicted climate variables

### 3. CNN-LSTM-GRU Model (`cnn_lstm_gru.py`)
- **Architecture**: CNN + LSTM + GRU hybrid
- **Features**:
  - Enhanced temporal modeling with GRU
  - Multi-scale feature extraction
  - Advanced attention mechanisms
- **Input**: Full sequence (10, height, width, channels)
- **Output**: Predicted climate variables

##  Performance Results

### Model Comparison
| Model | Temperature R² | Dewpoint R² | Heatwave RMSE | Coldwave RMSE |
|-------|----------------|-------------|---------------|---------------|
| CNN | 0.6782 | 0.6191 | 0.9179°C | 0.7343°C |
| CNN-LSTM | 0.4028 | 0.5041 | 1.3622°C | 0.7213°C |
| CNN-LSTM-GRU | 0.6337 | 0.5900 | 1.0279°C | 0.7083°C |

### Event Detection Performance
| Model | Heatwave F1 | Coldwave F1 | Heatwave Precision | Coldwave Precision |
|-------|-------------|-------------|-------------------|-------------------|
| CNN | 0.7834 | 0.8114 | 0.9430 | 0.9230 |
| CNN-LSTM | 0.8217 | 0.8342 | 0.8976 | 0.8351 |
| CNN-LSTM-GRU | 0.8438 | 0.8438 | 0.9038 | 0.8578 |

##  Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (for full dataset processing)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd heatwave-forecasting
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy pandas matplotlib seaborn scikit-learn xarray netCDF4 cartopy tqdm
   ```

4. **Verify GPU setup**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   nvidia-smi  # Check GPU status
   ```

##  Usage Guide

### 1. Data Preprocessing

```bash
# Process climate data with conservative approach (recommended)
python preprocess_data.py
```

**Configuration Options**:
- `sequence_length`: Number of input timesteps (default: 10)
- `forecast_horizon`: Prediction horizon (default: 1)
- `spatial_downsample`: Spatial resolution reduction (default: 2)
- `temporal_sample_ratio`: Temporal sampling ratio (default: 0.9)

### 2. Model Training

```bash
# Train CNN model
python Models/cnn_model.py

# Train CNN-LSTM model
python Models/cnn_lstm_model.py

# Train CNN-LSTM-GRU model
python Models/cnn_lstm_gru.py
```

**Training Parameters**:
- `batch_size`: 16 (optimized for GPU memory)
- `learning_rate`: 0.001
- `epochs`: 20
- `early_stopping`: Patience of 10 epochs

### 3. Model Evaluation

```bash
# Run comprehensive evaluation
python Models/final_report.py
```

This generates:
- Performance metrics comparison
- Event detection analysis
- Visualization plots
- Threshold optimization

### 4. GPU Monitoring

```bash
# Monitor GPU usage during training
bash check_gpu_usage.sh
```

##  Project Structure

```
heatwave-forecasting/
├── Models/                     # Model implementations
│   ├── cnn_model.py           # CNN architecture
│   ├── cnn_lstm_model.py      # CNN-LSTM hybrid
│   ├── cnn_lstm_gru.py        # CNN-LSTM-GRU hybrid
│   └── final_report.py        # Evaluation script
├── processed_climate_conservative/  # Processed data
│   ├── X_train.npz           # Training inputs
│   ├── y_train.npz           # Training targets
│   ├── X_val.npz             # Validation inputs
│   ├── y_val.npz             # Validation targets
│   ├── X_test.npz            # Test inputs
│   ├── y_test.npz            # Test targets
│   ├── scalers.pkl           # Data scalers
│   └── metadata.pkl          # Processing metadata
├── Trained_model/             # Saved model weights
├── Visulization/              # Generated plots and logs
├── samples/                   # Sample input data
├── backup/                    # Project notes and documentation
├── preprocess_data.py         # Data preprocessing pipeline
├── check_gpu_usage.sh         # GPU monitoring script
└── README.md                  # This file
```

## 🔧 Configuration

### Memory Management
- **Dataset size**: ~8GB for full conservative dataset
- **GPU memory**: 11GB+ recommended
- **RAM**: 16GB+ for processing

### Performance Optimization
- Use `spatial_downsample=4` for faster training
- Reduce `temporal_sample_ratio` for memory constraints
- Adjust `batch_size` based on GPU memory

##  Visualization

The project generates several visualization outputs:

1. **Model Metrics Comparison**: Bar charts comparing R², RMSE, MAE across models
2. **Event Detection Analysis**: Heatwave and coldwave detection performance
3. **Time Series Analysis**: Temporal patterns and predictions
4. **Seasonal Analysis**: Seasonal event detection patterns
5. **Spatial Maps**: Geographic distribution of predictions

##  Key Features

- **Multi-model comparison**: CNN, CNN-LSTM, CNN-LSTM-GRU architectures
- **Event detection**: Specialized heatwave and coldwave detection
- **Memory optimization**: Efficient data processing for large datasets
- **GPU acceleration**: CUDA support for fast training
- **Comprehensive evaluation**: Multiple metrics and visualizations
- **Threshold optimization**: Automatic optimal threshold finding

##  Technical Details

### Data Flow
```
NetCDF Files → Preprocessing → Sequence Creation → Model Training → Evaluation
```

### Model Input/Output
- **Input**: (batch_size, sequence_length, height, width, channels)
- **Output**: (batch_size, height, width, channels)
- **Channels**: [d2m, t2m] for both input and output

### Training Strategy
- **Loss function**: MSE for regression
- **Optimizer**: Adam with learning rate scheduling
- **Validation**: 15% of data for validation
- **Testing**: 15% of data for final evaluation

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Climate data provided by meteorological services
- PyTorch community for deep learning framework
- Scientific Python ecosystem for data processing tools

## 📞 Contact

For questions or issues, please open an issue on the repository or contact the development team.

---

**Note**: This project requires significant computational resources. Ensure you have adequate GPU memory and storage space before running the full pipeline. 
