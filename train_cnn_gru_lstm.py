import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import os
import json

# Import the model and loss function from the separate file
from models.cnn_gru_lstm import UltraWeatherModel, QuantileWeatherLossV2

# --- Configuration ---
HEATWAVE_CONFIG = {
    'seq_len': 7,  # Sequence length for input data
    'num_features': 13,  # Number of input features
    'hidden_dim': 256,
    'num_transformer_layers': 4,
    'num_heads': 8,
    'dropout': 0.2,
    'quantiles': [0.1, 0.5, 0.9],  # Predicting 10th, 50th (median), and 90th percentiles
    'learnable_pe': True,  # Enable learnable positional encoding
    'attention_l1_reg_weight': 1e-4,  # Small L1 regularization on attention weights
    'learning_rate': 0.001,
    'l2_reg_weight': 1e-5,  # L2 regularization (weight decay) for optimizer
    'num_epochs': 50,
    'batch_size': 64,
    'lr_scheduler_type': 'CosineAnnealingLR',  # Options: 'CosineAnnealingLR', 'ReduceLROnPlateau'
    'T_max': 50,  # For CosineAnnealingLR
    'patience': 10,  # For ReduceLROnPlateau
    'model_save_path': 'saved_models/ultra_weather_model.pth'
}

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create necessary directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('configs', exist_ok=True)

# --- Data Loading ---
print("Loading data...")
try:
    # Load and preprocess data
    x_train = np.load('data/X_train.npy').astype(np.float32)
    y_train = np.load('data/y_train.npy').astype(np.float32)
    x_val = np.load('data/X_val.npy').astype(np.float32)
    y_val = np.load('data/y_val.npy').astype(np.float32)
    
    # Reshape y data to (num_samples, 1) for consistency with quantile loss
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    # Generate day_of_year for seasonal encoding
    train_day_of_year = (np.arange(len(x_train)) % 365).astype(np.int64)
    val_day_of_year = (np.arange(len(x_val)) % 365).astype(np.int64)

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

except FileNotFoundError:
    print("Error: .npy files not found. Please ensure data files are in the correct location.")
    exit(1)

# --- PyTorch Datasets and DataLoaders ---
train_dataset = TensorDataset(
    torch.from_numpy(x_train),
    torch.from_numpy(y_train),
    torch.from_numpy(train_day_of_year)
)
val_dataset = TensorDataset(
    torch.from_numpy(x_val),
    torch.from_numpy(y_val),
    torch.from_numpy(val_day_of_year)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=HEATWAVE_CONFIG['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=HEATWAVE_CONFIG['batch_size'],
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# --- Model, Loss, Optimizer, Scheduler Initialization ---
model = UltraWeatherModel(
    seq_len=HEATWAVE_CONFIG['seq_len'],
    num_features=HEATWAVE_CONFIG['num_features'],
    hidden_dim=HEATWAVE_CONFIG['hidden_dim'],
    num_transformer_layers=HEATWAVE_CONFIG['num_transformer_layers'],
    num_heads=HEATWAVE_CONFIG['num_heads'],
    dropout=HEATWAVE_CONFIG['dropout'],
    num_quantiles=len(HEATWAVE_CONFIG['quantiles']),
    learnable_pe=HEATWAVE_CONFIG['learnable_pe'],
    attention_l1_reg_weight=HEATWAVE_CONFIG['attention_l1_reg_weight']
).to(device)

criterion = QuantileWeatherLossV2(
    quantiles=HEATWAVE_CONFIG['quantiles'],
    extreme_temp_threshold=35.0,
    cold_threshold=0.0,
    extreme_weight=3.0,
    seasonal_weight=1.5
).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=HEATWAVE_CONFIG['learning_rate'],
    weight_decay=HEATWAVE_CONFIG['l2_reg_weight']
)

# Learning Rate Scheduler
scheduler = None
if HEATWAVE_CONFIG['lr_scheduler_type'] == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(optimizer, T_max=HEATWAVE_CONFIG['T_max'])
elif HEATWAVE_CONFIG['lr_scheduler_type'] == 'ReduceLROnPlateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=HEATWAVE_CONFIG['patience'])

# --- Training Loop ---
print("\nStarting training...")
best_val_loss = float('inf')
best_val_r2 = float('-inf')
train_losses = []
val_losses = []

for epoch in range(HEATWAVE_CONFIG['num_epochs']):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y, batch_day_of_year in tqdm(train_loader, desc=f"Epoch {epoch+1}/{HEATWAVE_CONFIG['num_epochs']} Training"):
        batch_x, batch_y, batch_day_of_year = batch_x.to(device), batch_y.to(device), batch_day_of_year.to(device)
        
        optimizer.zero_grad()
        
        predictions, outputs = model(batch_x, batch_day_of_year)
        
        # Calculate loss including attention regularization
        loss = criterion(predictions, batch_y, batch_day_of_year, outputs.get('attention_reg_loss', 0.0))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # --- Validation Loop ---
    model.eval()
    val_loss = 0.0
    all_val_predictions = []
    all_val_targets = []

    with torch.no_grad():
        for batch_x, batch_y, batch_day_of_year in tqdm(val_loader, desc=f"Epoch {epoch+1}/{HEATWAVE_CONFIG['num_epochs']} Validation"):
            batch_x, batch_y, batch_day_of_year = batch_x.to(device), batch_y.to(device), batch_day_of_year.to(device)
            
            predictions, outputs = model(batch_x, batch_day_of_year)
            
            # Calculate loss including attention regularization
            loss = criterion(predictions, batch_y, batch_day_of_year, outputs.get('attention_reg_loss', 0.0))
            
            val_loss += loss.item() * batch_x.size(0)

            # For evaluation metrics, use the median (0.5 quantile) prediction
            median_idx = HEATWAVE_CONFIG['quantiles'].index(0.5)
            point_predictions = predictions[:, median_idx].cpu().numpy()
            
            all_val_predictions.extend(point_predictions)
            all_val_targets.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Calculate evaluation metrics
    val_mse = mean_squared_error(all_val_targets, all_val_predictions)
    val_mae = mean_absolute_error(all_val_targets, all_val_predictions)
    val_r2 = r2_score(all_val_targets, all_val_predictions)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}")

    # Step the learning rate scheduler
    if scheduler:
        if HEATWAVE_CONFIG['lr_scheduler_type'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:  # CosineAnnealingLR
            scheduler.step()

    # Save the best model based on R² score
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_r2': val_r2,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'config': HEATWAVE_CONFIG
        }, HEATWAVE_CONFIG['model_save_path'])
        print(f"\nNew best model saved! R²: {val_r2:.4f}")

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_r2': best_val_r2,
    'best_val_loss': best_val_loss
}

with open('configs/training_history.json', 'w') as f:
    json.dump(history, f, indent=4)

print("\nTraining complete.")