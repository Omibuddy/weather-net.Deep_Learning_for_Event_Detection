import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

# ----------------------------
# Data Loading and Preparation
# ----------------------------

class ClimateDataset(Dataset):
    """Dataset for climate data"""
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)['data']
        self.y = np.load(y_path)['data']
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        return x, y

def load_data():
    """Load and prepare test dataset"""
    X_path = "processed_climate_conservative/X_test.npz"
    y_path = "processed_climate_conservative/y_test.npz"
    
    # Load data from .npz files
    X_data = np.load(X_path)['data'].astype(np.float32)
    y_data = np.load(y_path)['data'].astype(np.float32)
    
    # Convert to tensors
    test_X = torch.tensor(X_data)
    test_y = torch.tensor(y_data)
    
    # Original shapes: 
    # test_X: (3576, 10, 51, 81, 2) -> [samples, timesteps, height, width, channels]
    # test_y: (3576, 1, 51, 81, 2)
    
    # Permute dimensions for target
    test_y_processed = test_y[:, 0].permute(0, 3, 1, 2)  # (3576, 2, 51, 81)
    
    # Prepare inputs for different models
    # For CNN: use last timestep but keep 5D format (batch, 1, height, width, channels)
    test_X_cnn = test_X[:, -1:, :, :, :]  # Last timestep with sequence dimension (3576, 1, 51, 81, 2)
    
    # For recurrent models: keep original format (batch, seq, height, width, channels)
    test_X_recurrent = test_X  # Keep original format (3576, 10, 51, 81, 2)
    
    return test_X_cnn, test_X_recurrent, test_y_processed

# ----------------------------
# Model Loading
# ----------------------------

def load_models(device):
    """Load pre-trained models from saved state dictionaries"""
    from Models.cnn_model import ClimateCNNModel
    from Models.cnn_lstm_model import CNN_LSTM_Model
    from Models.cnn_lstm_gru import CNN_LSTM_GRU_Model
    
    models = {}
    
    # Model parameters
    input_channels = 2
    output_channels = 2
    spatial_height = 51
    spatial_width = 81

    
    # Fixed model paths
    model_paths = {
        'CNN (loss)': 'Trained_model/cnn_model_best_loss.pth',
        'CNN (r2)': 'Trained_model/cnn_model_best_r2.pth',
        'CNN-LSTM (loss)': 'Trained_model/cnn_lstm_model_best_loss.pth',
        'CNN-LSTM (r2)': 'Trained_model/cnn_lstm_model_best_r2.pth',
        'CNN-LSTM-GRU (loss)': 'Trained_model/cnn_lstm_gru_best_loss.pth',
        'CNN-LSTM-GRU (r2)': 'Trained_model/cnn_lstm_gru_best_r2.pth'
    }
    
    
    # Load CNN models
    cnn_model_loss = ClimateCNNModel(input_channels, output_channels, spatial_height, spatial_width)
    cnn_model_loss.load_state_dict(torch.load(model_paths['CNN (loss)'], map_location=device))
    cnn_model_loss.to(device)
    cnn_model_loss.eval()
    models['CNN (loss)'] = cnn_model_loss
    
    cnn_model_r2 = ClimateCNNModel(input_channels, output_channels, spatial_height, spatial_width)
    cnn_model_r2.load_state_dict(torch.load(model_paths['CNN (r2)'], map_location=device))
    cnn_model_r2.to(device)
    cnn_model_r2.eval()
    models['CNN (r2)'] = cnn_model_r2
    
    # Load CNN-LSTM models
    cnnlstm_model_loss = CNN_LSTM_Model(input_channels, output_channels, 256, 2, spatial_height, spatial_width)
    cnnlstm_model_loss.load_state_dict(torch.load(model_paths['CNN-LSTM (loss)'], map_location=device))
    cnnlstm_model_loss.to(device)
    cnnlstm_model_loss.eval()
    models['CNN-LSTM (loss)'] = cnnlstm_model_loss
    
    cnnlstm_model_r2 = CNN_LSTM_Model(input_channels, output_channels, 256, 2, spatial_height, spatial_width)
    cnnlstm_model_r2.load_state_dict(torch.load(model_paths['CNN-LSTM (r2)'], map_location=device))
    cnnlstm_model_r2.to(device)
    cnnlstm_model_r2.eval()
    models['CNN-LSTM (r2)'] = cnnlstm_model_r2
    
    # Load CNN-LSTM-GRU models (using consistent parameters)
    cnnlstmgru_model_loss = CNN_LSTM_GRU_Model(input_channels, output_channels, 512, 3, spatial_height, spatial_width)
    cnnlstmgru_model_loss.load_state_dict(torch.load(model_paths['CNN-LSTM-GRU (loss)'], map_location=device), strict=False)
    cnnlstmgru_model_loss.to(device)
    cnnlstmgru_model_loss.eval()
    models['CNN-LSTM-GRU (loss)'] = cnnlstmgru_model_loss
    
    cnnlstmgru_model_r2 = CNN_LSTM_GRU_Model(input_channels, output_channels, 512, 3, spatial_height, spatial_width)
    cnnlstmgru_model_r2.load_state_dict(torch.load(model_paths['CNN-LSTM-GRU (r2)'], map_location=device), strict=False)
    cnnlstmgru_model_r2.to(device)
    cnnlstmgru_model_r2.eval()
    models['CNN-LSTM-GRU (r2)'] = cnnlstmgru_model_r2
    
    print("All models loaded successfully")
    return models

# ----------------------------
# Standardized Evaluation Functions
# ----------------------------

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics for temperature prediction"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # Calculate MAPE with proper handling of small values
    min_threshold = 1e-8
    valid_mask = np.abs(y_true_flat) > min_threshold
    
    if np.sum(valid_mask) > 0:
        y_true_valid = y_true_flat[valid_mask]
        y_pred_valid = y_pred_flat[valid_mask]
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / np.abs(y_true_valid))) * 100
    else:
        mape = np.nan
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def calculate_classification_metrics(true, pred, threshold, is_heatwave=True):
    """Calculate classification metrics for event detection using consistent threshold"""
    true_events = true > threshold if is_heatwave else true < threshold
    pred_events = pred > threshold if is_heatwave else pred < threshold
    
    TP = np.logical_and(true_events, pred_events).sum()
    FP = np.logical_and(~true_events, pred_events).sum()
    FN = np.logical_and(true_events, ~pred_events).sum()
    TN = np.logical_and(~true_events, ~pred_events).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }

def evaluate_models_standardized(models, test_X_cnn, test_X_recurrent, test_y, device):
    """Evaluate all models with standardized methodology"""
    # Extract true t2m values (channel index 1)
    true_t2m = test_y[:, 1, :, :].cpu().numpy()  # (3576, 51, 81)
    
    # Define global thresholds for fair comparison
    global_heat_threshold = np.percentile(true_t2m.flatten(), 90)
    global_cold_threshold = np.percentile(true_t2m.flatten(), 10)
    
    print(f"Global Heatwave Threshold (90th percentile): {global_heat_threshold:.4f}°C")
    print(f"Global Coldwave Threshold (10th percentile): {global_cold_threshold:.4f}°C")
    
    results = {}
    predictions = {}
    batch_size = 32
    
    with torch.no_grad():
        for name, model in models.items():
            print(f"Processing {name}...")
            
            # Initialize prediction array
            pred = np.zeros((3576, 51, 81))
            
            # Process in batches
            for i in range(0, 3576, batch_size):
                end_idx = min(i + batch_size, 3576)
                
                if 'CNN' in name and 'LSTM' not in name:
                    input_batch = test_X_cnn[i:end_idx].to(device)
                    output_batch = model(input_batch)
                else:
                    input_batch = test_X_recurrent[i:end_idx].to(device)
                    output_batch = model(input_batch)
                
                # Extract t2m predictions
                if len(output_batch.shape) == 4:
                    pred_batch = output_batch[:, :, :, 1].cpu().numpy()
                else:
                    pred_batch = output_batch.reshape(-1, 51, 81, 2)[:, :, :, 1].cpu().numpy()
                
                pred[i:end_idx] = pred_batch
            
            predictions[name] = pred
            
            # Calculate regression metrics
            regression_metrics = calculate_regression_metrics(true_t2m, pred)
            
            # Calculate classification metrics with global thresholds
            heat_metrics = calculate_classification_metrics(true_t2m, pred, global_heat_threshold, is_heatwave=True)
            cold_metrics = calculate_classification_metrics(true_t2m, pred, global_cold_threshold, is_heatwave=False)
            
            results[name] = {
                'Regression': regression_metrics,
                'Heatwave': heat_metrics,
                'Coldwave': cold_metrics
            }
    
    return results, predictions, true_t2m, global_heat_threshold, global_cold_threshold

# ----------------------------
# Visualization Functions
# ----------------------------

def plot_comprehensive_metrics(results):
    """Plot comprehensive evaluation metrics including both regression and classification"""
    # Prepare data for plotting
    model_names = list(results.keys())
    
    # Regression metrics
    r2_scores = [results[name]['Regression']['R2'] for name in model_names]
    rmse_scores = [results[name]['Regression']['RMSE'] for name in model_names]
    mae_scores = [results[name]['Regression']['MAE'] for name in model_names]
    
    # Classification metrics
    heat_f1_scores = [results[name]['Heatwave']['F1'] for name in model_names]
    cold_f1_scores = [results[name]['Coldwave']['F1'] for name in model_names]
    heat_precision = [results[name]['Heatwave']['Precision'] for name in model_names]
    cold_precision = [results[name]['Coldwave']['Precision'] for name in model_names]
    heat_recall = [results[name]['Heatwave']['Recall'] for name in model_names]
    cold_recall = [results[name]['Coldwave']['Recall'] for name in model_names]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Regression R² scores
    bars1 = axes[0, 0].bar(model_names, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[0, 0].set_title('Regression R² Scores', fontweight='bold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, r2_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: RMSE scores
    bars2 = axes[0, 1].bar(model_names, rmse_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[0, 1].set_title('RMSE Scores', fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (°C)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, rmse_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: MAE scores
    bars3 = axes[0, 2].bar(model_names, mae_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[0, 2].set_title('MAE Scores', fontweight='bold')
    axes[0, 2].set_ylabel('MAE (°C)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, mae_scores):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: F1 scores for both events
    x = np.arange(len(model_names))
    width = 0.35
    
    bars4a = axes[1, 0].bar(x - width/2, heat_f1_scores, width, label='Heatwave', color='#ff7f0e')
    bars4b = axes[1, 0].bar(x + width/2, cold_f1_scores, width, label='Coldwave', color='#1f77b4')
    axes[1, 0].set_title('F1 Scores for Event Detection', fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 5: Precision scores
    bars5a = axes[1, 1].bar(x - width/2, heat_precision, width, label='Heatwave', color='#ff7f0e')
    bars5b = axes[1, 1].bar(x + width/2, cold_precision, width, label='Coldwave', color='#1f77b4')
    axes[1, 1].set_title('Precision Scores', fontweight='bold')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    # Plot 6: Recall scores
    bars6a = axes[1, 2].bar(x - width/2, heat_recall, width, label='Heatwave', color='#ff7f0e')
    bars6b = axes[1, 2].bar(x + width/2, cold_recall, width, label='Coldwave', color='#1f77b4')
    axes[1, 2].set_title('Recall Scores', fontweight='bold')
    axes[1, 2].set_ylabel('Recall')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(model_names, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed results table
    print("\n" + "="*120)
    print("COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("="*120)
    
    print(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Heat F1':<10} {'Cold F1':<10} {'Heat Prec':<12} {'Cold Prec':<12}")
    print("-"*120)
    
    for name in model_names:
        reg = results[name]['Regression']
        heat = results[name]['Heatwave']
        cold = results[name]['Coldwave']
        
        print(f"{name:<20} {reg['R2']:<8.3f} {reg['RMSE']:<8.3f} {reg['MAE']:<8.3f} "
              f"{heat['F1']:<10.3f} {cold['F1']:<10.3f} {heat['Precision']:<12.3f} {cold['Precision']:<12.3f}")
    
    print("="*120)

def plot_event_day_standardized(true_t2m, predictions, global_heat_threshold, global_cold_threshold, event_name, day_idx):
    """Visualize event detection for a specific day with standardized thresholds"""
    num_models = len(predictions)
    fig, axes = plt.subplots(num_models+1, 3, figsize=(18, 4*(num_models+1)))
    plt.suptitle(f"{event_name} Detection - Day {day_idx} (Standardized Thresholds)", fontsize=16)
    
    # Use global thresholds for true events
    if event_name == "Heatwave":
        threshold = global_heat_threshold
        true_events = true_t2m[day_idx] > threshold
    else:
        threshold = global_cold_threshold
        true_events = true_t2m[day_idx] < threshold
    
    # Plot true data
    ax = axes[0, 0]
    im = ax.imshow(true_t2m[day_idx], cmap='coolwarm')
    ax.set_title("True Temperature")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    ax.imshow(true_events, cmap='binary')
    ax.set_title(f"True Events (Threshold: {threshold:.2f}°C)")
    
    # Leave third column empty for true data
    axes[0, 2].axis('off')
    
    # Plot model predictions
    for i, (model_name, pred) in enumerate(predictions.items(), start=1):
        # Temperature map
        ax = axes[i, 0]
        im = ax.imshow(pred[day_idx], cmap='coolwarm')
        ax.set_title(f"{model_name} Predicted Temperature")
        plt.colorbar(im, ax=ax)
        
        # Event detection map (using same global threshold)
        ax = axes[i, 1]
        pred_events = pred[day_idx] > threshold if event_name=="Heatwave" else pred[day_idx] < threshold
        ax.imshow(pred_events, cmap='binary')
        ax.set_title(f"{model_name} Events (Threshold: {threshold:.2f}°C)")
        
        # Error map (FP: white, FN: black, TP: gray)
        ax = axes[i, 2]
        error_map = np.zeros_like(true_events, dtype=float)
        error_map[pred_events & ~true_events] = 0.3  # FP = light gray
        error_map[~pred_events & true_events] = 0.6  # FN = dark gray
        error_map[pred_events & true_events] = 1.0    # TP = white
        ax.imshow(error_map, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{model_name} Error (FP/FN)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{event_name}_detection_day_{day_idx}_standardized.png', dpi=150)
    plt.close()

def plot_enhanced_time_series_standardized(true_t2m, predictions, global_heat_threshold, global_cold_threshold):
    """Enhanced time series visualization with standardized thresholds"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    plt.suptitle('Enhanced Event Detection Analysis (Standardized Thresholds)', fontsize=18)
    
    time_range = range(len(true_t2m))
    colors = plt.cm.tab10.colors
    model_names = list(predictions.keys())
    
    # Location to highlight
    lat, lon = 25, 40
    
    # Plot 1: Temperature with standardized thresholds
    axes[0].plot(time_range, true_t2m[:, lat, lon], 'k-', lw=2.5, label='True Temperature')
    
    for i, (name, pred) in enumerate(predictions.items()):
        # Plot temperature prediction
        axes[0].plot(time_range, pred[:, lat, lon], color=colors[i], 
                  alpha=0.8, lw=1.5, label=f'{name}')
    
    # Plot standardized thresholds
    axes[0].axhline(y=global_heat_threshold, color='red', linestyle='--', alpha=0.7, label='Heatwave Threshold')
    axes[0].axhline(y=global_cold_threshold, color='blue', linestyle=':', alpha=0.7, label='Coldwave Threshold')
    
    axes[0].set_title(f'Temperature at Location ({lat}, {lon}) with Standardized Thresholds', fontsize=14)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.2)
    
    # Plot 2: Model performance (F1 score) with standardized thresholds
    for i, name in enumerate(model_names):
        heat_f1 = []
        cold_f1 = []
        for t in time_range:
            # Heatwave F1 with standardized threshold
            heat_true = true_t2m[t] > global_heat_threshold
            heat_pred = predictions[name][t] > global_heat_threshold
            heat_f1.append(f1_score(heat_true.flatten(), heat_pred.flatten(), zero_division=0))
            
            # Coldwave F1 with standardized threshold
            cold_true = true_t2m[t] < global_cold_threshold
            cold_pred = predictions[name][t] < global_cold_threshold
            cold_f1.append(f1_score(cold_true.flatten(), cold_pred.flatten(), zero_division=0))
        
        axes[1].plot(time_range, heat_f1, color=colors[i], lw=1.5, 
                  label=f'{name} Heatwave')
        axes[1].plot(time_range, cold_f1, color=colors[i], lw=1.5, 
                  linestyle='--', label=f'{name} Coldwave')
    
    axes[1].set_title('Event Detection F1 Score Over Time (Standardized Thresholds)', fontsize=14)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.2)
    
    # Plot 3: Event detection accuracy with standardized thresholds
    for i, name in enumerate(model_names):
        accuracy = []
        for t in time_range:
            # Combined accuracy
            heat_correct = np.logical_and(
                true_t2m[t] > global_heat_threshold,
                predictions[name][t] > global_heat_threshold
            )
            cold_correct = np.logical_and(
                true_t2m[t] < global_cold_threshold,
                predictions[name][t] < global_cold_threshold
            )
            normal_correct = np.logical_and(
                np.logical_and(true_t2m[t] <= global_heat_threshold, true_t2m[t] >= global_cold_threshold),
                np.logical_and(predictions[name][t] <= global_heat_threshold, predictions[name][t] >= global_cold_threshold)
            )
            
            total_correct = np.sum(heat_correct) + np.sum(cold_correct) + np.sum(normal_correct)
            total_pixels = heat_correct.size
            accuracy.append(total_correct / total_pixels)
        
        axes[2].plot(time_range, accuracy, color=colors[i], lw=1.5, label=name)
    
    axes[2].set_title('Event Detection Accuracy Over Time (Standardized Thresholds)', fontsize=14)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_ylim(0.5, 1)
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, alpha=0.2)
    
    # Plot 4: Event intensity difference with standardized thresholds
    for i, name in enumerate(model_names):
        intensity_diff = []
        for t in time_range:
            # Heatwave intensity difference
            heat_true = np.where(true_t2m[t] > global_heat_threshold, 
                               true_t2m[t] - global_heat_threshold, 0)
            heat_pred = np.where(predictions[name][t] > global_heat_threshold, 
                               predictions[name][t] - global_heat_threshold, 0)
            heat_diff = np.abs(heat_true - heat_pred).mean()
            
            # Coldwave intensity difference
            cold_true = np.where(true_t2m[t] < global_cold_threshold, 
                               global_cold_threshold - true_t2m[t], 0)
            cold_pred = np.where(predictions[name][t] < global_cold_threshold, 
                               global_cold_threshold - predictions[name][t], 0)
            cold_diff = np.abs(cold_true - cold_pred).mean()
            
            intensity_diff.append((heat_diff + cold_diff) / 2)
        
        axes[3].plot(time_range, intensity_diff, color=colors[i], 
                  lw=1.5, label=name)
    
    axes[3].set_title('Average Event Intensity Error (Standardized Thresholds)', fontsize=14)
    axes[3].set_xlabel('Time (days)', fontsize=12)
    axes[3].set_ylabel('Intensity Error (°C)', fontsize=12)
    axes[3].legend(loc='upper right', fontsize=10)
    axes[3].grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('enhanced_time_series_analysis_standardized.png', dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------
# Main Execution
# ----------------------------

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    test_X_cnn, test_X_recurrent, test_y = load_data()
    
    # Move data to device
    test_X_cnn = test_X_cnn.to(device)
    test_X_recurrent = test_X_recurrent.to(device)
    test_y = test_y.to(device)
    
    # Load models
    print("Loading models...")
    models = load_models(device)
    
    # Evaluate models with standardized methodology
    print("Evaluating models with standardized methodology...")
    results, predictions, true_t2m, global_heat_threshold, global_cold_threshold = evaluate_models_standardized(
        models, test_X_cnn, test_X_recurrent, test_y, device
    )
    
    # Print and plot comprehensive results
    plot_comprehensive_metrics(results)
    
    # Find extreme event days using global thresholds
    heat_extreme_day = np.argmax((true_t2m > global_heat_threshold).mean(axis=(1, 2)))
    cold_extreme_day = np.argmax((true_t2m < global_cold_threshold).mean(axis=(1, 2)))
    
    print(f"Heatwave extreme day index: {heat_extreme_day}")
    print(f"Coldwave extreme day index: {cold_extreme_day}")
    
    # Generate visualizations with standardized thresholds
    print("Generating visualizations...")
    plot_event_day_standardized(true_t2m, predictions, global_heat_threshold, global_cold_threshold, "Heatwave", heat_extreme_day)
    plot_event_day_standardized(true_t2m, predictions, global_heat_threshold, global_cold_threshold, "Coldwave", cold_extreme_day)
    plot_enhanced_time_series_standardized(true_t2m, predictions, global_heat_threshold, global_cold_threshold)
    
    # Print standardized thresholds
    print("\nStandardized Thresholds:")
    print(f"  Heatwave threshold (90th percentile): {global_heat_threshold:.4f}°C")
    print(f"  Coldwave threshold (10th percentile): {global_cold_threshold:.4f}°C")
    
    # Find best performing models
    print("\nBest Performing Models:")
    
    # Best regression model
    best_regression = max(results.keys(), key=lambda x: results[x]['Regression']['R2'])
    print(f"  Best Regression (R²): {best_regression} (R² = {results[best_regression]['Regression']['R2']:.3f})")
    
    # Best heatwave detection
    best_heatwave = max(results.keys(), key=lambda x: results[x]['Heatwave']['F1'])
    print(f"  Best Heatwave Detection: {best_heatwave} (F1 = {results[best_heatwave]['Heatwave']['F1']:.3f})")
    
    # Best coldwave detection
    best_coldwave = max(results.keys(), key=lambda x: results[x]['Coldwave']['F1'])
    print(f"  Best Coldwave Detection: {best_coldwave} (F1 = {results[best_coldwave]['Coldwave']['F1']:.3f})")
    
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()