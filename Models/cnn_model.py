import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

# --- Early Stopping Class ---
class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    Monitors either validation loss (minimize) or R² score (maximize).
    """
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss', mode='min'):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            monitor (str): Metric to monitor ('val_loss' or 'val_r2')
            mode (str): 'min' for loss, 'max' for R²
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, current_score, epoch):
        """
        Check if training should stop.
        
        Args:
            current_score (float): Current value of monitored metric
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = current_score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False

# --- 1. CNN Model Definition ---
class ClimateCNNModel(nn.Module):
    def __init__(self, input_channels, output_channels, spatial_height, spatial_width):
        """
        Initializes the CNN model for climate prediction.
        Args:
            input_channels (int): Number of input channels (2 for 'd2m' and 't2m')
            output_channels (int): Number of output channels (2 for predicted variables)
            spatial_height (int): Height of spatial dimensions
            spatial_width (int): Width of spatial dimensions
        """
        super(ClimateCNNModel, self).__init__()
        
        # --- Enhanced Feature Extraction ---
        self.features = nn.Sequential(
            # Block 1 - Reduced complexity
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # Reduced from 64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Block 2 - Reduced complexity
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced from 128
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Block 3 - Reduced complexity
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Reduced from 256
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate size after convolutions
        self._to_linear = None
        self._calculate_conv_output_size(spatial_height, spatial_width)
        
        # --- Regression Head ---
        self.regressor = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, output_channels * spatial_height * spatial_width)
        )
        
        # Store output dimensions
        self.output_channels = output_channels
        self.output_height = spatial_height
        self.output_width = spatial_width

    def _calculate_conv_output_size(self, h, w):
        """Calculate the output size after convolutional layers"""
        # Create a dummy tensor to calculate feature size
        dummy = torch.zeros(1, self.features[0].in_channels, h, w)
        dummy_out = self.features(dummy)
        self._to_linear = int(np.prod(dummy_out.size()[1:]))
        return self._to_linear

    def forward(self, x):
        # Input shape: (batch_size, seq_len, height, width, channels)
        # We'll use the last time step only for CNN
        last_time_step = x[:, -1, :, :, :]
        
        # Permute to (batch_size, channels, height, width)
        x_permuted = last_time_step.permute(0, 3, 1, 2)
        
        # Feature extraction
        features = self.features(x_permuted)
        
        # Flatten features - use reshape instead of view for better compatibility
        features = features.reshape(features.size(0), -1)
        
        # Regression
        output = self.regressor(features)
        
        # Reshape to spatial format
        output = output.reshape(-1, self.output_height, self.output_width, self.output_channels)
        return output

# --- 2. Data Loading and Preparation ---
def load_data(data_dir):
    try:
        print(f"Loading data from {data_dir}...")
        
        # Check if files are .npz (compressed) or .npy (regular)
        train_file = os.path.join(data_dir, 'X_train.npz')
        if os.path.exists(train_file):
            # Load compressed .npz files
            print("Loading compressed .npz files...")
            
            # Load training data
            X_train_data = np.load(os.path.join(data_dir, 'X_train.npz'))
            y_train_data = np.load(os.path.join(data_dir, 'y_train.npz'))
            
            # Get the array names
            X_train_key = X_train_data.files[0]
            y_train_key = y_train_data.files[0]
            
            X_train = X_train_data[X_train_key]
            y_train = y_train_data[y_train_key]
            print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # Load validation data
            X_val_data = np.load(os.path.join(data_dir, 'X_val.npz'))
            y_val_data = np.load(os.path.join(data_dir, 'y_val.npz'))
            
            X_val_key = X_val_data.files[0]
            y_val_key = y_val_data.files[0]
            
            X_val = X_val_data[X_val_key]
            y_val = y_val_data[y_val_key]
            print(f"Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            # Load test data
            X_test_data = np.load(os.path.join(data_dir, 'X_test.npz'))
            y_test_data = np.load(os.path.join(data_dir, 'y_test.npz'))
            
            X_test_key = X_test_data.files[0]
            y_test_key = y_test_data.files[0]
            
            X_test = X_test_data[X_test_key]
            y_test = y_test_data[y_test_key]
            print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
            
        else:
            # Load regular .npy files
            print("Loading regular .npy files...")
            
            # Load training data
            X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
            y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
            print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # Load validation data
            X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
            y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
            print(f"Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            # Load test data
            X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
            y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
            print(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Load metadata if available
        metadata_path = os.path.join(data_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print("Loaded metadata:", metadata)
        
        print("Data loaded successfully!")
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in {data_dir}")
        print(f"Error details: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

# --- 3. Model Training ---
def train_model():
    # Load the preprocessed data
    data_dir = 'processed_climate_conservative'
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)

    if X_train is None:
        print("Failed to load data. Exiting...")
        return

    # Print data shapes
    print(f"\nOriginal data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Reshape target data
    y_train = y_train.squeeze(1)
    y_val = y_val.squeeze(1)
    y_test = y_test.squeeze(1)

    print(f"\nAfter reshaping:")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 32  # Larger batch size possible for CNN
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"\nUsing batch size: {batch_size}")

    # Initialize the CNN model
    input_channels = X_train.shape[-1]
    output_channels = y_train.shape[-1]
    spatial_height = X_train.shape[2]
    spatial_width = X_train.shape[3]
    
    print(f"\nModel parameters:")
    print(f"input_channels: {input_channels}, output_channels: {output_channels}")
    print(f"spatial_height: {spatial_height}, spatial_width: {spatial_width}")
    
    model = ClimateCNNModel(
        input_channels=input_channels,
        output_channels=output_channels,
        spatial_height=spatial_height,
        spatial_width=spatial_width
    )

    # Use GPU if available
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training setup
    num_epochs = 50
    best_val_loss = float('inf')
    best_r2 = -float('inf')

    # Early stopping setup - can be configured to monitor 'val_loss' or 'val_r2'
    early_stopping_monitor = 'val_loss'  # Options: 'val_loss' or 'val_r2'
    early_stopping_patience = 15
    early_stopping_min_delta = 0.001
    
    if early_stopping_monitor == 'val_loss':
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            monitor='val_loss',
            mode='min'
        )
    else:  # 'val_r2'
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            monitor='val_r2',
            mode='max'
        )

    # Setup logging for overfitting analysis
    log_file = 'cnn_training_log.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tTrain_R2\tVal_R2\n")

    print("\n" + "="*120)
    print("🚀 CNN MODEL TRAINING STARTED")
    print("="*120)
    print(f"Early stopping: monitoring {early_stopping_monitor} with patience={early_stopping_patience}")
    print(f"Training log will be saved to: {log_file}")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train R²':<10} {'Val R²':<10} "
          f"{'Train RMSE':<12} {'Val RMSE':<12} {'Train MAE':<10} {'Val MAE':<10}")
    print("-"*120)

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_targets = []

        # Training phase - simplified without detailed progress bar
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(labels.detach().cpu().numpy())

        # Validation phase - simplified without detailed progress bar
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        # Calculate metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_pred = np.concatenate(train_predictions, axis=0)
        train_true = np.concatenate(train_targets, axis=0)
        val_pred = np.concatenate(val_predictions, axis=0)
        val_true = np.concatenate(val_targets, axis=0)

        train_metrics = calculate_metrics(train_true, train_pred)
        val_metrics = calculate_metrics(val_true, val_pred)
        
        # Update learning rate
        scheduler.step(avg_val_loss)

        # Print epoch summary - single line per epoch
        print(f"{epoch+1:<6} {avg_train_loss:<12.4f} {avg_val_loss:<12.4f} {train_metrics['R2']:<10.4f} "
              f"{val_metrics['R2']:<10.4f} {train_metrics['RMSE']:<12.4f} {val_metrics['RMSE']:<12.4f} "
              f"{train_metrics['MAE']:<10.4f} {val_metrics['MAE']:<10.4f}")

        # Log metrics for overfitting analysis
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\t{train_metrics['R2']:.6f}\t{val_metrics['R2']:.6f}\n")

        # Save best models
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'cnn_model_best_loss.pth')
            print(f"  → Saved best loss model: {avg_val_loss:.4f}")
            
        if val_metrics['R2'] > best_r2:
            best_r2 = val_metrics['R2']
            torch.save(model.state_dict(), 'cnn_model_best_r2.pth')
            print(f"  → Saved best R² model: {best_r2:.4f}")

        # Early stopping check
        if early_stopping_monitor == 'val_loss':
            should_stop = early_stopping(avg_val_loss, epoch + 1)
        else:  # 'val_r2'
            should_stop = early_stopping(val_metrics['R2'], epoch + 1)
            
        if should_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"Best {early_stopping_monitor} was {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
            break

    print("-"*120)
    print("✅ TRAINING COMPLETED")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"🏆 Best validation R²: {best_r2:.4f}")
    if early_stopping.early_stop:
        print(f"🛑 Training stopped early at epoch {epoch + 1}")
        print(f"Best {early_stopping_monitor}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
    print("="*120)
    
    # --- Model Evaluation ---
    print("\n\n" + "="*60)
    print("🧪 MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load('cnn_model_best_loss.pth'))
    model.eval()
    
    # Prepare test data
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    test_predictions = []
    test_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(labels.cpu().numpy())
    
    # Calculate test metrics
    avg_test_loss = test_loss / len(test_loader)
    test_pred = np.concatenate(test_predictions, axis=0)
    test_true = np.concatenate(test_targets, axis=0)
    test_metrics = calculate_metrics(test_true, test_pred)
    
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test R²: {test_metrics['R2']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test MAE: {test_metrics['MAE']:.4f}")
    print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")
    
    # Detailed error analysis
    print(f"\n" + "="*60)
    print("🔍 DETAILED ERROR ANALYSIS")
    print("="*60)
    
    error_stats = test_metrics['error_stats']
    print(f"Total predictions: {error_stats['total_predictions']:,}")
    print(f"Valid predictions (for MAPE): {error_stats['valid_predictions']:,}")
    print(f"Filtered out (too small): {error_stats['filtered_out']:,}")
    
    if not np.isnan(error_stats['mean_abs_error']):
        print(f"\nAbsolute Error Statistics:")
        print(f"  Mean Absolute Error: {error_stats['mean_abs_error']:.4f}")
        print(f"  Median Absolute Error: {error_stats['median_abs_error']:.4f}")
        print(f"  Std Absolute Error: {error_stats['std_abs_error']:.4f}")
        
        print(f"\nRelative Error Statistics:")
        print(f"  Mean Relative Error: {error_stats['mean_rel_error']:.2f}%")
        print(f"  Median Relative Error: {error_stats['median_rel_error']:.2f}%")
        print(f"  Std Relative Error: {error_stats['std_rel_error']:.2f}%")
        print(f"  Min Relative Error: {error_stats['min_rel_error']:.2f}%")
        print(f"  Max Relative Error: {error_stats['max_rel_error']:.2f}%")
        
        # Target value distribution analysis
        print(f"\nTarget Value Distribution:")
        print(f"  Min target value: {np.min(test_true):.6f}")
        print(f"  Max target value: {np.max(test_true):.6f}")
        print(f"  Mean target value: {np.mean(test_true):.6f}")
        print(f"  Median target value: {np.median(test_true):.6f}")
        print(f"  Std target value: {np.std(test_true):.6f}")
        
        # Check for very small values that might cause MAPE issues
        small_values = np.sum(np.abs(test_true) < 0.01)
        zero_values = np.sum(test_true == 0)
        print(f"  Values < 0.01: {small_values:,} ({small_values/len(test_true)*100:.1f}%)")
        print(f"  Zero values: {zero_values:,} ({zero_values/len(test_true)*100:.1f}%)")
        
        # Prediction vs Actual analysis
        print(f"\nPrediction vs Actual Analysis:")
        print(f"  Min prediction: {np.min(test_pred):.6f}")
        print(f"  Max prediction: {np.max(test_pred):.6f}")
        print(f"  Mean prediction: {np.mean(test_pred):.6f}")
        print(f"  Median prediction: {np.median(test_pred):.6f}")
        
        # Bias analysis
        bias = np.mean(test_pred - test_true)
        print(f"  Prediction bias: {bias:.6f} (positive = overestimation)")
        
        # Correlation analysis
        correlation = np.corrcoef(test_true.flatten(), test_pred.flatten())[0, 1]
        print(f"  Correlation (pred vs actual): {correlation:.4f}")
        
    else:
        print("⚠️  No valid predictions for MAPE calculation (all values too small)")
    
    print("="*60)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics with improved MAPE handling and error analysis"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Improved MAPE calculation with better handling of edge cases
    # Filter out values that are too small to avoid division by zero issues
    min_threshold = 1e-8  # Minimum threshold to avoid division by very small numbers
    valid_mask = np.abs(y_true_flat) > min_threshold
    
    if np.sum(valid_mask) > 0:
        y_true_valid = y_true_flat[valid_mask]
        y_pred_valid = y_pred_flat[valid_mask]
        
        # Calculate MAPE only on valid values
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / np.abs(y_true_valid))) * 100
        
        # Additional error analysis
        abs_errors = np.abs(y_true_valid - y_pred_valid)
        rel_errors = np.abs((y_true_valid - y_pred_valid) / np.abs(y_true_valid))
        
        # Statistics for error analysis
        error_stats = {
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors),
            'std_abs_error': np.std(abs_errors),
            'mean_rel_error': np.mean(rel_errors) * 100,
            'median_rel_error': np.median(rel_errors) * 100,
            'std_rel_error': np.std(rel_errors) * 100,
            'max_rel_error': np.max(rel_errors) * 100,
            'min_rel_error': np.min(rel_errors) * 100,
            'valid_predictions': len(y_true_valid),
            'total_predictions': len(y_true_flat),
            'filtered_out': len(y_true_flat) - len(y_true_valid)
        }
    else:
        mape = np.nan
        error_stats = {
            'mean_abs_error': np.nan,
            'median_abs_error': np.nan,
            'std_abs_error': np.nan,
            'mean_rel_error': np.nan,
            'median_rel_error': np.nan,
            'std_rel_error': np.nan,
            'max_rel_error': np.nan,
            'min_rel_error': np.nan,
            'valid_predictions': 0,
            'total_predictions': len(y_true_flat),
            'filtered_out': len(y_true_flat)
        }
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'error_stats': error_stats
    }

if __name__ == '__main__':
    train_model()