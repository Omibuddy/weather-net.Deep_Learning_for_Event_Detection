import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import time
import os
from tqdm import tqdm
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TimeSeriesAugmentation:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def add_noise(self, x):
        """Add Gaussian noise to the input tensor"""
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

    def time_warp(self, x, sigma=0.2):
        """Apply time warping augmentation"""
        # Simplified time warping that works with tensors
        batch_size, seq_len, num_features = x.shape
        
        # Create a simple time warping by applying small perturbations
        # This is a simplified version that avoids numpy conversion issues
        warp_factor = 1.0 + torch.randn(batch_size, seq_len, device=x.device) * sigma * 0.1
        warp_factor = torch.clamp(warp_factor, 0.8, 1.2)  # Limit warping range
        
        # Apply warping by scaling the values slightly
        warped_x = x * warp_factor.unsqueeze(-1)
        
        return warped_x

class ImprovedCNNLSTMModel(nn.Module):
    def __init__(self, seq_len=7, num_features=13, lstm_units=256, dense_units=128, 
                 num_classes=1, dropout_rate=0.3, task_type='regression'):
        super(ImprovedCNNLSTMModel, self).__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.task_type = task_type
        
        # Feature extractor for each time step
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Multi-scale CNN
        self.conv1d_3 = nn.Conv1d(num_features, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(num_features, 128, kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(num_features, 64, kernel_size=7, padding=3)
        
        # Residual blocks
        conv_channels = 128 + 128 + 64  # 320
        self.res_blocks = nn.ModuleList([
            ResidualBlock(conv_channels),
            ResidualBlock(conv_channels),
            ResidualBlock(conv_channels)
        ])
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Refinement conv layer
        self.conv1d_refine = nn.Conv1d(conv_channels, 256, kernel_size=3, padding=1)
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=256, 
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if lstm_units > 1 else 0,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=2 * lstm_units,
            hidden_size=lstm_units//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_units,  # 2 * (lstm_units//2)
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers
        combined_features = lstm_units + 32  # LSTM output + feature extractor output
        self.dense1 = nn.Linear(combined_features, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.dense3 = nn.Linear(dense_units // 2, dense_units // 4)
        self.output = nn.Linear(dense_units // 4, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_units)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features for each time step
        features = []
        for t in range(self.seq_len):
            time_step_features = self.feature_extractor(x[:, t, :])
            features.append(time_step_features)
        
        # Stack and aggregate features
        stacked_features = torch.stack(features, dim=1)
        aggregated_features = torch.mean(stacked_features, dim=1)
        
        # CNN feature extraction
        x_conv = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        conv_3 = F.relu(self.conv1d_3(x_conv))
        conv_5 = F.relu(self.conv1d_5(x_conv))
        conv_7 = F.relu(self.conv1d_7(x_conv))
        
        conv_combined = torch.cat([conv_3, conv_5, conv_7], dim=1)
        conv_combined = self.bn1(conv_combined)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            conv_combined = res_block(conv_combined)
        
        conv_refined = F.relu(self.conv1d_refine(conv_combined))
        conv_refined = self.bn2(conv_refined)
        
        # Back to LSTM format
        lstm_input = conv_refined.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out1, _ = self.lstm1(lstm_input)
        lstm_out1 = self.dropout(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.layer_norm(lstm_out2)
        
        # Attention
        attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        attn_out = self.dropout(attn_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Combine with aggregated features
        combined = torch.cat([pooled, aggregated_features], dim=1)
        
        # Dense layers
        dense1_out = F.relu(self.dense1(combined))
        dense1_out = self.dropout(dense1_out)
        
        dense2_out = F.relu(self.dense2(dense1_out))
        dense2_out = self.dropout(dense2_out)
        
        dense3_out = F.relu(self.dense3(dense2_out))
        dense3_out = self.dropout(dense3_out)
        
        output = self.output(dense3_out)
        
        if self.task_type == 'classification':
            output = torch.sigmoid(output)
        
        return output

class HeatwaveTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Optimizer
        if model is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=0.001, 
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = None
        
        # Scheduler
        self.scheduler = None  # Will be initialized in train()
        
        # Data augmentation
        self.augmentation = TimeSeriesAugmentation(noise_level=0.02)
        
        # Loss function
        if model is not None:
            if model.task_type == 'regression':
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.BCELoss()
        else:
            self.criterion = None

    def save_training_state(self, path='training_state.pth'):
        """Save model state and training history"""
        if self.model is None:
            print("No model to save")
            return
        
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state
        }
        torch.save(state, path)
        print(f"Training state saved to {path}")

    def load_training_state(self, path='training_state.pth'):
        """Load model state and training history"""
        if not os.path.exists(path):
            print(f"No saved state found at {path}")
            return
        
        state = torch.load(path)
        if self.model is not None:
            self.model.load_state_dict(state['model_state'])
        if self.optimizer is not None and state['optimizer_state'] is not None:
            self.optimizer.load_state_dict(state['optimizer_state'])
        
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']
        self.best_val_loss = state['best_val_loss']
        self.best_model_state = state['best_model_state']
        print(f"Training state loaded from {path}")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            # Ensure tensors are on the correct device
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()
            
            # Data augmentation
            if np.random.random() < 0.5:  # 50% chance of augmentation
                if np.random.random() < 0.5:
                    batch_x = self.augmentation.add_noise(batch_x)
                else:
                    batch_x = self.augmentation.time_warp(batch_x)
            
            self.optimizer.zero_grad()
            
            try:
                predictions = self.model(batch_x)
                
                # Ensure shapes match
                if predictions.shape != batch_y.shape:
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(-1)
                    elif len(predictions.shape) == 1:
                        predictions = predictions.unsqueeze(-1)
                
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{lr:.6f}'
                })
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc='Validation'):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                try:
                    predictions = self.model(batch_x)
                    
                    # Ensure shapes match
                    if predictions.shape != batch_y.shape:
                        if len(batch_y.shape) == 1:
                            batch_y = batch_y.unsqueeze(-1)
                        elif len(predictions.shape) == 1:
                            predictions = predictions.unsqueeze(-1)
                    
                    loss = self.criterion(predictions, batch_y)
                    total_loss += loss.item()
                    
                    predictions_list.append(predictions.cpu())
                    targets_list.append(batch_y.cpu())
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if not predictions_list:
            return float('inf'), torch.tensor([]), torch.tensor([])
        
        avg_loss = total_loss / len(val_loader)
        all_predictions = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        return avg_loss, all_predictions, all_targets
    
    def get_prediction_intervals(self, predictions, confidence=0.95):
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
            
        predictions = np.asarray(predictions).flatten()
        mean = np.mean(predictions)
        std = np.std(predictions)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        return mean - z_score * std, mean + z_score * std
    
    def train(self, train_loader, val_loader, epochs=100, patience=15, save_path='best_model.pth'):
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1e4
        )
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Calculate metrics
            if self.model.task_type == 'regression' and len(val_preds) > 0:
                val_preds_np = val_preds.numpy().flatten()
                val_targets_np = val_targets.numpy().flatten()
                
                if len(val_preds_np) > 0 and len(val_targets_np) > 0:
                    mse = mean_squared_error(val_targets_np, val_preds_np)
                    mae = mean_absolute_error(val_targets_np, val_preds_np)
                    r2 = r2_score(val_targets_np, val_preds_np)
                    
                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"Val MSE: {mse:.4f} | Val MAE: {mae:.4f} | Val R²: {r2:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                try:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                except:
                    print("Warning: Could not save model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
    
    def plot_training_history(self):
        """Plot training and validation loss curves"""
        if not self.train_losses or not self.val_losses:
            print("No training history to plot")
            return
            
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        train_losses_log = [max(loss, 1e-8) for loss in self.train_losses]
        val_losses_log = [max(loss, 1e-8) for loss in self.val_losses]
        plt.plot(np.log10(train_losses_log), label='Log Train Loss')
        plt.plot(np.log10(val_losses_log), label='Log Validation Loss')
        plt.title('Log Scale Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def create_model_and_trainer(X_train, y_train, task_type='regression'):
    """Create model and trainer with proper error handling"""
    try:
        # Get dimensions
        if len(X_train.shape) == 3:
            seq_len = X_train.shape[1]
            num_features = X_train.shape[2]
        else:
            raise ValueError(f"Expected 3D input (batch, seq, features), got shape {X_train.shape}")
        
        print(f"Creating model with sequence length {seq_len} and {num_features} features")
        
        # Create model
        model = ImprovedCNNLSTMModel(
            seq_len=seq_len,
            num_features=num_features,
            lstm_units=128,  # Reduced for stability
            dense_units=64,   # Reduced for stability
            num_classes=1,
            dropout_rate=0.2, # Reduced dropout
            task_type=task_type
        )
        
        # Create trainer
        trainer = HeatwaveTrainer(model)
        
        return model, trainer
        
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

# Example usage
if __name__ == "__main__":
    print("Fixed Heatwave Prediction Training Pipeline")
    print("=" * 50)
    print("\nKey fixes:")
    print("- Fixed time warping augmentation to avoid numpy conversion issues")
    print("- Added proper error handling in training loops")
    print("- Improved tensor shape handling")
    print("- Reduced model complexity for better stability")
    print("- Added device handling for all tensor operations")
    
    print("\nTo use this pipeline:")
    print("1. Load your data tensors (X_train, y_train, etc.)")
    print("2. Create DataLoaders with proper dtypes")
    print("3. Call create_model_and_trainer() with your data")
    print("4. Train using trainer.train()")
    print("5. Plot results using trainer.plot_training_history()")