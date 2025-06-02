import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from models.cnn_gru_lstm import UltraWeatherModel
from tqdm import tqdm

def load_test_data(batch_size=32):
    """Load and preprocess test data"""
    print("Loading test data...")
    
    try:
        # Load numpy arrays
        X_test = np.load('data/X_test.npy').astype(np.float32)
        y_test = np.load('data/y_test.npy').astype(np.float32)
        
        # Reshape y data to (num_samples, 1) for consistency with quantile loss
        y_test = y_test.reshape(-1, 1)
        
        # Generate day_of_year for seasonal encoding
        test_day_of_year = (np.arange(len(X_test)) % 365).astype(np.int64)
        
        print(f"Test data shape: {X_test.shape}")
        
        # Convert to PyTorch tensors
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        test_day_of_year = torch.from_numpy(test_day_of_year)
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(X_test, y_test, test_day_of_year)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return test_loader, y_test.numpy()
        
    except FileNotFoundError:
        print("Error: Test data files not found. Please ensure 'data/X_test.npy' and 'data/y_test.npy' exist.")
        exit(1)

def load_model(model_path, config):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UltraWeatherModel(
        seq_len=config['seq_len'],
        num_features=config['num_features'],
        hidden_dim=config['hidden_dim'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_quantiles=len(config['quantiles']),
        learnable_pe=config['learnable_pe'],
        attention_l1_reg_weight=config['attention_l1_reg_weight']
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_predictions(y_true, y_pred, y_pred_lower, y_pred_upper):
    """Calculate evaluation metrics including prediction intervals"""
    # Calculate metrics for median prediction
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate prediction interval coverage
    coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Prediction_Interval_Coverage': coverage
    }
    
    return metrics

def save_predictions(y_true, y_pred, y_pred_lower, y_pred_upper, save_path):
    """Save predictions to CSV file including prediction intervals"""
    results_df = pd.DataFrame({
        'True_Values': y_true.squeeze(),
        'Predictions': y_pred.squeeze(),
        'Lower_Bound': y_pred_lower.squeeze(),
        'Upper_Bound': y_pred_upper.squeeze()
    })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def main():
    # Load configuration
    with open('configs/training_config.json', 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_loader, y_test = load_test_data(batch_size=config['batch_size'])
    
    # Load model
    model = load_model('saved_models/ultra_weather_model.pth', config)
    
    # Make predictions
    print("Making predictions...")
    predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_day_of_year in tqdm(test_loader, desc='Predicting'):
            batch_x = batch_x.to(device)
            batch_day_of_year = batch_day_of_year.to(device)
            batch_pred, _ = model(batch_x, batch_day_of_year)
            predictions.append(batch_pred.cpu().numpy())
    
    # Process predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Extract median prediction and prediction intervals
    median_idx = config['quantiles'].index(0.5)
    lower_idx = config['quantiles'].index(0.1)
    upper_idx = config['quantiles'].index(0.9)
    
    y_pred = predictions[:, median_idx]
    y_pred_lower = predictions[:, lower_idx]
    y_pred_upper = predictions[:, upper_idx]
    
    # Evaluate predictions
    metrics = evaluate_predictions(y_test, y_pred, y_pred_lower, y_pred_upper)
    
    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save predictions
    save_predictions(
        y_test,
        y_pred,
        y_pred_lower,
        y_pred_upper,
        'results/processed/cnn_gru_lstm_test_predictions.csv'
    )

if __name__ == "__main__":
    main()