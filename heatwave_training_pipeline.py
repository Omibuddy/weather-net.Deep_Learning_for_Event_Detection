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
warnings.filterwarnings('ignore')

class ImprovedCNNLSTMModel(nn.Module):
    def __init__(self, seq_len=30, num_features=5, lstm_units=128, dense_units=64, 
                 num_classes=1, dropout_rate=0.3, task_type='regression'):
        super(ImprovedCNNLSTMModel, self).__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.task_type = task_type
        
        # Multi-scale CNN for temporal pattern extraction
        self.conv1d_3 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.conv1d_7 = nn.Conv1d(num_features, 32, kernel_size=7, padding=3)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(160)  # 64 + 64 + 32
        self.bn2 = nn.BatchNorm1d(128)
        
        # Additional conv layer
        self.conv1d_refine = nn.Conv1d(160, 128, kernel_size=3, padding=1)
        
        # Stacked bidirectional LSTM
        self.lstm1 = nn.LSTM(
            input_size=128, 
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=2 * lstm_units,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * lstm_units,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers
        self.dense1 = nn.Linear(2 * lstm_units, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.output = nn.Linear(dense_units // 2, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(2 * lstm_units)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x_conv = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        conv_3 = F.relu(self.conv1d_3(x_conv))
        conv_5 = F.relu(self.conv1d_5(x_conv))
        conv_7 = F.relu(self.conv1d_7(x_conv))
        
        conv_combined = torch.cat([conv_3, conv_5, conv_7], dim=1)
        conv_combined = self.bn1(conv_combined)
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
        
        # Dense layers
        dense1_out = F.relu(self.dense1(pooled))
        dense1_out = self.dropout(dense1_out)
        
        dense2_out = F.relu(self.dense2(dense1_out))
        dense2_out = self.dropout(dense2_out)
        
        output = self.output(dense2_out)
        
        if self.task_type == 'classification':
            output = torch.sigmoid(output)
        
        return output

class HeatwaveTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5,
            min_lr=1e-6
        )
        
        # Loss function
        if model.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            
            # Reshape if needed
            if predictions.shape != batch_y.shape:
                batch_y = batch_y.view(predictions.shape)
            
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc='Validation'):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                
                if predictions.shape != batch_y.shape:
                    batch_y = batch_y.view(predictions.shape)
                
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                
                predictions_list.append(predictions.cpu())
                targets_list.append(batch_y.cpu())
        
        avg_loss = total_loss / len(val_loader)
        all_predictions = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        return avg_loss, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=100, patience=15, save_path='best_model.pth'):
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate metrics
            if self.model.task_type == 'regression':
                mse = mean_squared_error(val_targets.numpy(), val_preds.numpy())
                mae = mean_absolute_error(val_targets.numpy(), val_preds.numpy())
                r2 = r2_score(val_targets.numpy(), val_preds.numpy())
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Val MSE: {mse:.4f} | Val MAE: {mae:.4f} | Val R²: {r2:.4f}")
            else:
                val_preds_binary = (val_preds > 0.5).float()
                accuracy = (val_preds_binary == val_targets).float().mean()
                try:
                    auc = roc_auc_score(val_targets.numpy(), val_preds.numpy())
                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"Val Accuracy: {accuracy:.4f} | Val AUC: {auc:.4f}")
                except:
                    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                    print(f"Val Accuracy: {accuracy:.4f}")
            
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, save_path)
                patience_counter = 0
                print("✓ New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        return self.train_losses, self.val_losses
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc='Testing'):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                
                if predictions.shape != batch_y.shape:
                    batch_y = batch_y.view(predictions.shape)
                
                predictions_list.append(predictions.cpu())
                targets_list.append(batch_y.cpu())
        
        all_predictions = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        return all_predictions, all_targets
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-50:], label='Training Loss (Last 50)', alpha=0.8)
        plt.plot(self.val_losses[-50:], label='Validation Loss (Last 50)', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_model_and_trainer(X_train, y_train, task_type='regression'):
    """Create model and trainer based on your data dimensions"""
    
    # Get dimensions from your data
    seq_len = X_train.shape[1]  # Time steps
    num_features = X_train.shape[2]  # Features
    
    print(f"Data dimensions:")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of features: {num_features}")
    print(f"- Task type: {task_type}")
    
    # Determine output dimension
    if len(y_train.shape) == 1:
        num_classes = 1
    else:
        num_classes = y_train.shape[1]
    
    # Model parameters
    model_params = {
        'seq_len': seq_len,
        'num_features': num_features,
        'lstm_units': 128,
        'dense_units': 64,
        'num_classes': num_classes,
        'dropout_rate': 0.3,
        'task_type': task_type
    }
    
    # Create model
    model = ImprovedCNNLSTMModel(**model_params)
    
    # Create trainer
    trainer = HeatwaveTrainer(model)
    
    return model, trainer, model_params

# Example usage with your data
if __name__ == "__main__":
    # Assuming you have X_train, y_train, X_val, y_val, X_test, y_test as tensors
    # and train_loader, val_loader, test_loader as DataLoaders
    
    print("Heatwave Prediction Training Pipeline")
    print("=" * 50)
    
    # Example of how to use with your existing data
    # Replace these with your actual data loading
    """
    # Create model and trainer
    model, trainer, model_params = create_model_and_trainer(X_train, y_train, task_type='regression')
    
    print(f"Model created with parameters: {model_params}")
    
    # Train the model
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        patience=15,
        save_path='heatwave_model.pth'
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    test_predictions, test_targets = trainer.evaluate(test_loader)
    
    # Calculate final metrics
    if model.task_type == 'regression':
        test_mse = mean_squared_error(test_targets.numpy(), test_predictions.numpy())
        test_mae = mean_absolute_error(test_targets.numpy(), test_predictions.numpy())
        test_r2 = r2_score(test_targets.numpy(), test_predictions.numpy())
        
        print(f"\nFinal Test Results:")
        print(f"MSE: {test_mse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"R²: {test_r2:.4f}")
    else:
        test_accuracy = ((test_predictions > 0.5).float() == test_targets).float().mean()
        test_auc = roc_auc_score(test_targets.numpy(), test_predictions.numpy())
        
        print(f"\nFinal Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"AUC: {test_auc:.4f}")
    """
    
    print("\nTo use this pipeline:")
    print("1. Load your data tensors (X_train, y_train, etc.)")
    print("2. Create DataLoaders")
    print("3. Call create_model_and_trainer() with your data")
    print("4. Train using trainer.train()")
    print("5. Evaluate using trainer.evaluate()")
