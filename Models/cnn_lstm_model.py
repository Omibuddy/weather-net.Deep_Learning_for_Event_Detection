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

# --- 1. Improved CNN-LSTM Model Definition with Anti-Overfitting Techniques ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size, num_layers, spatial_height, spatial_width, dropout_rate=0.3):
        """
        Initializes the CNN-LSTM model with anti-overfitting techniques.
        Args:
            input_channels (int): The number of input channels (2 for 'd2m' and 't2m').
            output_channels (int): The number of output channels (2 for the predicted variables).
            hidden_size (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of recurrent layers in the LSTM.
            spatial_height (int): Height of the spatial dimensions.
            spatial_width (int): Width of the spatial dimensions.
            dropout_rate (float): Dropout rate for regularization (default: 0.3).
        """
        super(CNN_LSTM_Model, self).__init__()
        
        # --- CNN Feature Extractor with Dropout and Reduced Capacity ---
        self.cnn = nn.Sequential(
            # First conv layer - reduced channels
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout for early layers
            
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout_rate * 0.7),
            
            # Third conv layer - reduced from 256 to 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout_rate),
            
            # Additional conv layer for better feature extraction
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout_rate)
        )
        
        # Adaptive pooling to reduce spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --- LSTM with Dropout ---
        # Reduced hidden size from 512 to 256
        self.lstm = nn.LSTM(
            input_size=128,  # Reduced from 256
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0  # Dropout between LSTM layers
        )

        # --- Output Layer with Dropout ---
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_channels * spatial_height * spatial_width)

        # Store output dimensions for reshaping
        self.output_channels = output_channels
        self.output_height = spatial_height
        self.output_width = spatial_width

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, height, width, channels).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, height, width, channels).
        """
        batch_size, seq_len, h, w, c = x.size()
        
        # Reshape to process all time steps together
        x_reshaped = x.permute(0, 1, 4, 2, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size * seq_len, c, h, w)

        # Pass each time step through the CNN
        c_out = self.cnn(x_reshaped)
        c_out = self.avgpool(c_out)
        c_out = c_out.view(batch_size, seq_len, -1)

        # Pass the sequence of features through the LSTM
        lstm_out, _ = self.lstm(c_out)

        # We are interested in the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Apply dropout before final layer
        last_time_step_out = self.dropout(last_time_step_out)

        # Pass the LSTM output through the fully connected layer
        output = self.fc(last_time_step_out)

        # Reshape the output to the desired image format
        output = output.view(batch_size, h, w, self.output_channels)

        return output

# --- Learning Rate Scheduler ---
class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# --- Data Augmentation ---
def augment_data(inputs, labels, noise_factor=0.01):
    """
    Simple data augmentation for spatial data - only noise to avoid dimension issues.
    """
    # Add small random noise
    if np.random.random() < 0.5:
        noise = torch.randn_like(inputs) * noise_factor
        inputs = inputs + noise
    
    return inputs, labels

# --- 2. Data Loading and Preparation ---
# Updated to load data from processed_climate_data folder

def load_data(data_dir):
    """
    Loads the preprocessed data from the processed_climate_data folder.
    Args:
        data_dir (str): The path to the processed_climate_data directory.
    Returns:
        tuple: A tuple containing the training, validation, and testing data.
    """
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
            
            # Get the array names (usually 'arr_0' for single arrays)
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
            print("Loading regular .npz files...")
            
            # Load training data
            X_train = np.load(os.path.join(data_dir, 'X_train.npz'))
            y_train = np.load(os.path.join(data_dir, 'y_train.npz'))
            print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
            
            # Load validation data
            X_val = np.load(os.path.join(data_dir, 'X_val.npz'))
            y_val = np.load(os.path.join(data_dir, 'y_val.npz'))
            print(f"Loaded X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            # Load test data
            X_test = np.load(os.path.join(data_dir, 'X_test.npz'))
            y_test = np.load(os.path.join(data_dir, 'y_test.npz'))
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
# This is the main function that orchestrates the model training process.

def train_model():
    """
    Main function to train the CNN-LSTM model with anti-overfitting techniques.
    """
    # Load the preprocessed CNN-LSTM data from processed_climate_data folder
    data_dir = 'processed_climate_conservative'
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_dir)

    if X_train is None:
        print("Failed to load data. Exiting...") 
        return # Exit if data loading failed

    # Print data shapes for debugging
    print(f"Original data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")

    # Reshape target data to match model output format
    y_train = y_train.squeeze(1)
    y_val = y_val.squeeze(1)
    y_test = y_test.squeeze(1)

    print(f"After reshaping:")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    # Create TensorDatasets and DataLoaders for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Reduced batch size for better generalization
    batch_size = 8  # Reduced from 16
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Using batch size: {batch_size}")

    # Initialize the model with anti-overfitting parameters
    input_channels = X_train.shape[-1]
    output_channels = y_train.shape[-1]
    hidden_size = 256  # Reduced from 512
    num_layers = 2
    dropout_rate = 0.3  # Added dropout
    
    print(f"Model parameters:")
    print(f"input_channels: {input_channels}")
    print(f"output_channels: {output_channels}")
    print(f"hidden_size: {hidden_size}")
    print(f"dropout_rate: {dropout_rate}")
    
    # Get spatial dimensions from the data
    spatial_height = X_train.shape[2]
    spatial_width = X_train.shape[3]
    
    print(f"spatial_height: {spatial_height}")
    print(f"spatial_width: {spatial_width}")
    
    model = CNN_LSTM_Model(
        input_channels=input_channels, 
        output_channels=output_channels, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        spatial_height=spatial_height, 
        spatial_width=spatial_width,
        dropout_rate=dropout_rate
    )

    # Use GPU if available, otherwise CPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    
    model.to(device)

    # Loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.0005,  # Reduced learning rate
        weight_decay=1e-4,  # Added weight decay for regularization
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    num_epochs = 50  # Increased epochs since we have early stopping
    warmup_epochs = 5
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, min_lr=1e-6)

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_r2 = -float('inf')

    # Early stopping setup
    early_stopping_monitor = 'val_loss'
    early_stopping_patience = 10  # Reduced patience
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
    log_file = 'cnn_lstm_training_log.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tTrain_R2\tVal_R2\tLR\n")

    print("\n" + "="*100)
    print("🚀 TRAINING STARTED WITH ANTI-OVERFITTING TECHNIQUES")
    print("="*100)
    print(f"Early stopping: monitoring {early_stopping_monitor} with patience={early_stopping_patience}")
    print(f"Training log will be saved to: {log_file}")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train R²':<10} {'Val R²':<10} {'LR':<10}")
    print("-"*100)

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_targets = []

        # Training loop with data augmentation
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Apply data augmentation during training
            inputs, labels = augment_data(inputs, labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(labels.detach().cpu().numpy())

        # Update learning rate
        current_lr = scheduler.step(epoch)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        # Calculate average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Concatenate all predictions and targets
        train_pred = np.concatenate(train_predictions, axis=0)
        train_true = np.concatenate(train_targets, axis=0)
        val_pred = np.concatenate(val_predictions, axis=0)
        val_true = np.concatenate(val_targets, axis=0)

        # Calculate metrics
        train_metrics = calculate_metrics(train_true, train_pred)
        val_metrics = calculate_metrics(val_true, val_pred)

        # Print epoch results
        print(f"{epoch+1:<6} {avg_train_loss:<12.4f} {avg_val_loss:<12.4f} {train_metrics['R2']:<10.4f} {val_metrics['R2']:<10.4f} {current_lr:<10.6f}")

        # Log metrics for overfitting analysis
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\t{train_metrics['R2']:.6f}\t{val_metrics['R2']:.6f}\t{current_lr:.6f}\n")

        # Save model based on validation loss and R² score
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'cnn_lstm_model_best_loss.pth')
            print(f"  → Model saved (best loss): {avg_val_loss:.4f}")
            
        if val_metrics['R2'] > best_r2:
            best_r2 = val_metrics['R2']
            torch.save(model.state_dict(), 'cnn_lstm_model_best_r2.pth')
            print(f"  → Model saved (best R²): {best_r2:.4f}")

        # Early stopping check
        if early_stopping_monitor == 'val_loss':
            should_stop = early_stopping(avg_val_loss, epoch + 1)
        else:  # 'val_r2'
            should_stop = early_stopping(val_metrics['R2'], epoch + 1)
            
        if should_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"Best {early_stopping_monitor} was {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
            break

    print("-"*100)
    print("✅ TRAINING COMPLETED")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"🏆 Best validation R²: {best_r2:.4f}")
    if early_stopping.early_stop:
        print(f"🛑 Training stopped early at epoch {epoch + 1}")
        print(f"Best {early_stopping_monitor}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
    print("="*100)

def calculate_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics with improved MAPE handling and error analysis.
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
    
    Returns:
        dict: Dictionary containing all metrics and error statistics
    """
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
    # Make sure you have the 'processed_climate_data' folder with the required .npy files.
    train_model()
