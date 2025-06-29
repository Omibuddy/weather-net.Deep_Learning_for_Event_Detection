import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
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
    from cnn_model import ClimateCNNModel
    from cnn_lstm_model import CNN_LSTM_Model
    from cnn_lstm_gru import CNN_LSTM_GRU_Model
    
    models = {}
    
    # Model parameters
    input_channels = 2
    output_channels = 2
    spatial_height = 51
    spatial_width = 81
    hidden_size = 256
    num_layers = 2
    
    # Load CNN models
    cnn_model_loss = ClimateCNNModel(input_channels, output_channels, spatial_height, spatial_width)
    cnn_model_loss.load_state_dict(torch.load('model/cnn_model_best_loss.pth', map_location=device))
    cnn_model_loss.to(device)
    cnn_model_loss.eval()
    models['CNN (loss)'] = cnn_model_loss
    
    cnn_model_r2 = ClimateCNNModel(input_channels, output_channels, spatial_height, spatial_width)
    cnn_model_r2.load_state_dict(torch.load('model/cnn_model_best_r2.pth', map_location=device))
    cnn_model_r2.to(device)
    cnn_model_r2.eval()
    models['CNN (r2)'] = cnn_model_r2
    
    # Load CNN-LSTM models
    cnnlstm_model_loss = CNN_LSTM_Model(input_channels, output_channels, hidden_size, num_layers, spatial_height, spatial_width)
    cnnlstm_model_loss.load_state_dict(torch.load('model/cnn_lstm_model_best_loss.pth', map_location=device))
    cnnlstm_model_loss.to(device)
    cnnlstm_model_loss.eval()
    models['CNN-LSTM (loss)'] = cnnlstm_model_loss
    
    cnnlstm_model_r2 = CNN_LSTM_Model(input_channels, output_channels, hidden_size, num_layers, spatial_height, spatial_width)
    cnnlstm_model_r2.load_state_dict(torch.load('model/cnn_lstm_model_best_r2.pth', map_location=device))
    cnnlstm_model_r2.to(device)
    cnnlstm_model_r2.eval()
    models['CNN-LSTM (r2)'] = cnnlstm_model_r2
    
    # Load CNN-LSTM-GRU models
    cnnlstmgru_model_loss = CNN_LSTM_GRU_Model(input_channels, output_channels, 512, 3, spatial_height, spatial_width)
    cnnlstmgru_model_loss.load_state_dict(torch.load('model/cnn_lstm_gru_best_loss.pth', map_location=device), strict=False)
    cnnlstmgru_model_loss.to(device)
    cnnlstmgru_model_loss.eval()
    models['CNN-LSTM-GRU (loss)'] = cnnlstmgru_model_loss
    
    cnnlstmgru_model_r2 = CNN_LSTM_GRU_Model(input_channels, output_channels, 512, 3, spatial_height, spatial_width)
    cnnlstmgru_model_r2.load_state_dict(torch.load('model/cnn_lstm_gru_best_r2.pth', map_location=device), strict=False)
    cnnlstmgru_model_r2.to(device)
    cnnlstmgru_model_r2.eval()
    models['CNN-LSTM-GRU (r2)'] = cnnlstmgru_model_r2
    
    print("All models loaded successfully")
    return models

# ----------------------------
# Threshold Optimization
# ----------------------------

def find_optimal_threshold(true, pred, event_type, n_thresholds=100):
    """Find optimal threshold for event detection using F1 score"""
    # Flatten arrays for grid search
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    
    best_threshold = None
    best_f1 = -1
    
    if event_type == "heatwave":
        # Search between 80th and 99th percentiles
        thresholds = np.linspace(np.percentile(true, 80), np.percentile(true, 99), n_thresholds)
        for thresh in thresholds:
            true_events = true_flat > thresh
            pred_events = pred_flat > thresh
            f1 = f1_score(true_events, pred_events, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
    else:  # coldwave
        # Search between 1st and 20th percentiles
        thresholds = np.linspace(np.percentile(true, 1), np.percentile(true, 20), n_thresholds)
        for thresh in thresholds:
            true_events = true_flat < thresh
            pred_events = pred_flat < thresh
            f1 = f1_score(true_events, pred_events, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
    return best_threshold

# ----------------------------
# Evaluation Functions
# ----------------------------

def calculate_metrics(true, pred, threshold, is_heatwave=True):
    """Calculate precision, recall, and F1 score for event detection"""
    true_events = true > threshold if is_heatwave else true < threshold
    pred_events = pred > threshold if is_heatwave else pred < threshold
    
    TP = np.logical_and(true_events, pred_events).sum()
    FP = np.logical_and(~true_events, pred_events).sum()
    FN = np.logical_and(true_events, ~pred_events).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_models(models, test_X_cnn, test_X_recurrent, test_y, device):
    """Evaluate all models on test data with batching and optimized thresholds"""
    # Extract true t2m values (channel index 1)
    true_t2m = test_y[:, 1, :, :].cpu().numpy()  # (3576, 51, 81)
    
    results = {}
    predictions = {}
    best_thresholds = {}
    batch_size = 32  # Reduce batch size
    
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
            
            # Find optimal thresholds for this model
            heat_threshold = find_optimal_threshold(true_t2m, pred, "heatwave")
            cold_threshold = find_optimal_threshold(true_t2m, pred, "coldwave")
            best_thresholds[name] = {'Heatwave': heat_threshold, 'Coldwave': cold_threshold}
            
            # Calculate metrics
            heat_precision, heat_recall, heat_f1 = calculate_metrics(
                true_t2m, pred, heat_threshold, is_heatwave=True
            )
            cold_precision, cold_recall, cold_f1 = calculate_metrics(
                true_t2m, pred, cold_threshold, is_heatwave=False
            )
            
            results[name] = {
                'Heatwave': {
                    'Precision': heat_precision, 
                    'Recall': heat_recall, 
                    'F1': heat_f1,
                    'Threshold': heat_threshold
                },
                'Coldwave': {
                    'Precision': cold_precision, 
                    'Recall': cold_recall, 
                    'F1': cold_f1,
                    'Threshold': cold_threshold
                }
            }
    
    return results, predictions, true_t2m, best_thresholds

# ----------------------------
# Visualization Functions
# ----------------------------

def plot_metrics(results):
    """Plot evaluation metrics as a table"""
    metrics = []
    for model, events in results.items():
        for event, scores in events.items():
            metrics.append({
                'Model': model,
                'Event': event,
                'Precision': scores['Precision'],
                'Recall': scores['Recall'],
                'F1 Score': scores['F1'],
                'Threshold': scores['Threshold']
            })
    
    df = pd.DataFrame(metrics)
    print("\nModel Performance Metrics:")
    print(df.to_string(index=False))
    
    # Plot metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('Model Performance Comparison', fontsize=16, pad=20)
    plt.savefig('model_metrics_comparison.png', bbox_inches='tight')
    plt.close()

def plot_event_day(true_t2m, predictions, best_thresholds, event_name, day_idx):
    """Visualize event detection for a specific day with model-specific thresholds"""
    num_models = len(predictions)
    fig, axes = plt.subplots(num_models+1, 3, figsize=(18, 4*(num_models+1)))
    plt.suptitle(f"{event_name} Detection - Day {day_idx}", fontsize=16)
    
    # Compute true events using true data's 90/10 percentile
    if event_name == "Heatwave":
        global_threshold = np.percentile(true_t2m.flatten(), 90)
        true_events = true_t2m[day_idx] > global_threshold
    else:
        global_threshold = np.percentile(true_t2m.flatten(), 10)
        true_events = true_t2m[day_idx] < global_threshold
    
    # Plot true data
    ax = axes[0, 0]
    im = ax.imshow(true_t2m[day_idx], cmap='coolwarm')
    ax.set_title("True Temperature")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    ax.imshow(true_events, cmap='binary')
    ax.set_title(f"True Events (Threshold: {global_threshold:.2f}°C)")
    
    # Leave third column empty for true data
    axes[0, 2].axis('off')
    
    # Plot model predictions
    for i, (model_name, pred) in enumerate(predictions.items(), start=1):
        # Get model-specific threshold
        threshold = best_thresholds[model_name][event_name]
        
        # Temperature map
        ax = axes[i, 0]
        im = ax.imshow(pred[day_idx], cmap='coolwarm')
        ax.set_title(f"{model_name} Predicted Temperature")
        plt.colorbar(im, ax=ax)
        
        # Event detection map
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
    plt.savefig(f'{event_name}_detection_day_{day_idx}.png', dpi=150)
    plt.close()

def plot_enhanced_time_series(true_t2m, predictions, best_thresholds):
    """Enhanced time series visualization with dynamic thresholds"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    plt.suptitle('Enhanced Event Detection Analysis', fontsize=18)
    
    time_range = range(len(true_t2m))
    colors = plt.cm.tab10.colors
    model_names = list(predictions.keys())
    
    # Location to highlight
    lat, lon = 25, 40
    
    # Plot 1: Temperature with dynamic thresholds
    axes[0].plot(time_range, true_t2m[:, lat, lon], 'k-', lw=2.5, label='True Temperature')
    
    for i, (name, pred) in enumerate(predictions.items()):
        # Get model-specific thresholds
        heat_thresh = best_thresholds[name]['Heatwave']
        cold_thresh = best_thresholds[name]['Coldwave']
        
        # Plot temperature prediction
        axes[0].plot(time_range, pred[:, lat, lon], color=colors[i], 
                  alpha=0.8, lw=1.5, label=f'{name}')
        
        # Plot thresholds
        axes[0].axhline(y=heat_thresh, color=colors[i], linestyle='--', alpha=0.7)
        axes[0].axhline(y=cold_thresh, color=colors[i], linestyle=':', alpha=0.7)
    
    axes[0].set_title(f'Temperature at Location ({lat}, {lon}) with Model-Specific Thresholds', fontsize=14)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.2)
    
    # Plot 2: Model performance (F1 score)
    for i, name in enumerate(model_names):
        heat_f1 = []
        cold_f1 = []
        for t in time_range:
            # Heatwave F1
            heat_true = true_t2m[t] > best_thresholds[name]['Heatwave']
            heat_pred = predictions[name][t] > best_thresholds[name]['Heatwave']
            heat_f1.append(f1_score(heat_true.flatten(), heat_pred.flatten(), zero_division=0))
            
            # Coldwave F1
            cold_true = true_t2m[t] < best_thresholds[name]['Coldwave']
            cold_pred = predictions[name][t] < best_thresholds[name]['Coldwave']
            cold_f1.append(f1_score(cold_true.flatten(), cold_pred.flatten(), zero_division=0))
        
        axes[1].plot(time_range, heat_f1, color=colors[i], lw=1.5, 
                  label=f'{name} Heatwave')
        axes[1].plot(time_range, cold_f1, color=colors[i], lw=1.5, 
                  linestyle='--', label=f'{name} Coldwave')
    
    axes[1].set_title('Event Detection F1 Score Over Time', fontsize=14)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.2)
    
    # Plot 3: Event detection accuracy
    for i, name in enumerate(model_names):
        accuracy = []
        for t in time_range:
            # Combined accuracy
            heat_correct = np.logical_and(
                true_t2m[t] > best_thresholds[name]['Heatwave'],
                predictions[name][t] > best_thresholds[name]['Heatwave']
            ) | np.logical_and(
                true_t2m[t] < best_thresholds[name]['Coldwave'],
                predictions[name][t] < best_thresholds[name]['Coldwave']
            )
            normal_correct = np.logical_and(
                np.logical_and(
                    true_t2m[t] >= best_thresholds[name]['Coldwave'],
                    true_t2m[t] <= best_thresholds[name]['Heatwave']
                ),
                np.logical_and(
                    predictions[name][t] >= best_thresholds[name]['Coldwave'],
                    predictions[name][t] <= best_thresholds[name]['Heatwave']
                )
            )
            accuracy.append((heat_correct.sum() + normal_correct.sum()) / true_t2m[t].size)
        
        axes[2].plot(time_range, accuracy, color=colors[i], 
                  lw=1.5, label=name)
    
    axes[2].set_title('Event Detection Accuracy', fontsize=14)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_ylim(0.5, 1)
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, alpha=0.2)
    
    # Plot 4: Event intensity difference
    for i, name in enumerate(model_names):
        intensity_diff = []
        for t in time_range:
            # Heatwave intensity difference
            heat_true = np.where(true_t2m[t] > best_thresholds[name]['Heatwave'], 
                               true_t2m[t] - best_thresholds[name]['Heatwave'], 0)
            heat_pred = np.where(predictions[name][t] > best_thresholds[name]['Heatwave'], 
                               predictions[name][t] - best_thresholds[name]['Heatwave'], 0)
            heat_diff = np.abs(heat_true - heat_pred).mean()
            
            # Coldwave intensity difference
            cold_true = np.where(true_t2m[t] < best_thresholds[name]['Coldwave'], 
                               best_thresholds[name]['Coldwave'] - true_t2m[t], 0)
            cold_pred = np.where(predictions[name][t] < best_thresholds[name]['Coldwave'], 
                               best_thresholds[name]['Coldwave'] - predictions[name][t], 0)
            cold_diff = np.abs(cold_true - cold_pred).mean()
            
            intensity_diff.append((heat_diff + cold_diff) / 2)
        
        axes[3].plot(time_range, intensity_diff, color=colors[i], 
                  lw=1.5, label=name)
    
    axes[3].set_title('Average Event Intensity Error', fontsize=14)
    axes[3].set_xlabel('Time (days)', fontsize=12)
    axes[3].set_ylabel('Intensity Error (°C)', fontsize=12)
    axes[3].legend(loc='upper right', fontsize=10)
    axes[3].grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('enhanced_time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_event_analysis(true_t2m, predictions, best_thresholds):
    """Plot seasonal analysis of heatwave and coldwave events"""
    # Assuming 30 days per month
    days_per_month = 30
    num_months = len(true_t2m) // days_per_month
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seasonal Event Detection Analysis', fontsize=16)
    
    colors = plt.cm.tab10.colors
    model_names = list(predictions.keys())
    
    # Prepare data structures
    monthly_heat_coverage = {name: [] for name in model_names}
    monthly_cold_coverage = {name: [] for name in model_names}
    monthly_heat_f1 = {name: [] for name in model_names}
    monthly_cold_f1 = {name: [] for name in model_names}
    
    # True data metrics
    true_heat_coverage = []
    true_cold_coverage = []
    
    for month in range(num_months):
        start_idx = month * days_per_month
        end_idx = min((month + 1) * days_per_month, len(true_t2m))
        
        # Process true data
        month_true = true_t2m[start_idx:end_idx]
        true_heat_coverage.append((month_true > np.percentile(true_t2m.flatten(), 90)).mean() * 100)
        true_cold_coverage.append((month_true < np.percentile(true_t2m.flatten(), 10)).mean() * 100)
        
        # Process model predictions
        for name in model_names:
            pred = predictions[name][start_idx:end_idx]
            heat_thresh = best_thresholds[name]['Heatwave']
            cold_thresh = best_thresholds[name]['Coldwave']
            
            # Coverage
            heat_coverage = (pred > heat_thresh).mean() * 100
            cold_coverage = (pred < cold_thresh).mean() * 100
            monthly_heat_coverage[name].append(heat_coverage)
            monthly_cold_coverage[name].append(cold_coverage)
            
            # F1 scores
            heat_true = month_true > heat_thresh
            heat_pred = pred > heat_thresh
            heat_f1 = f1_score(heat_true.flatten(), heat_pred.flatten(), zero_division=0)
            
            cold_true = month_true < cold_thresh
            cold_pred = pred < cold_thresh
            cold_f1 = f1_score(cold_true.flatten(), cold_pred.flatten(), zero_division=0)
            
            monthly_heat_f1[name].append(heat_f1)
            monthly_cold_f1[name].append(cold_f1)
    
    months = range(1, num_months + 1)
    
    # Plot 1: Monthly heatwave coverage
    ax = axes[0, 0]
    ax.plot(months, true_heat_coverage, 'k-', lw=3, label='True')
    for i, name in enumerate(model_names):
        ax.plot(months, monthly_heat_coverage[name], color=colors[i], lw=1.5, label=name)
    ax.set_title('Monthly Heatwave Coverage', fontsize=14)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # Plot 2: Monthly coldwave coverage
    ax = axes[0, 1]
    ax.plot(months, true_cold_coverage, 'k-', lw=3, label='True')
    for i, name in enumerate(model_names):
        ax.plot(months, monthly_cold_coverage[name], color=colors[i], lw=1.5, label=name)
    ax.set_title('Monthly Coldwave Coverage', fontsize=14)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # Plot 3: Monthly heatwave F1 scores
    ax = axes[1, 0]
    for i, name in enumerate(model_names):
        ax.plot(months, monthly_heat_f1[name], color=colors[i], lw=2, marker='o', label=name)
    ax.set_title('Monthly Heatwave F1 Scores', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # Plot 4: Monthly coldwave F1 scores
    ax = axes[1, 1]
    for i, name in enumerate(model_names):
        ax.plot(months, monthly_cold_f1[name], color=colors[i], lw=2, marker='o', label=name)
    ax.set_title('Monthly Coldwave F1 Scores', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('seasonal_event_analysis.png', dpi=300, bbox_inches='tight')
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
    
    # Evaluate models with optimized thresholds
    print("Evaluating models with optimized thresholds...")
    results, predictions, true_t2m, best_thresholds = evaluate_models(
        models, test_X_cnn, test_X_recurrent, test_y, device
    )
    
    # Print and plot results
    plot_metrics(results)
    
    # Find extreme event days using true data's global thresholds
    global_heat_threshold = np.percentile(true_t2m.flatten(), 90)
    global_cold_threshold = np.percentile(true_t2m.flatten(), 10)
    
    heat_extreme_day = np.argmax((true_t2m > global_heat_threshold).mean(axis=(1, 2)))
    cold_extreme_day = np.argmax((true_t2m < global_cold_threshold).mean(axis=(1, 2)))
    
    print(f"Heatwave extreme day index: {heat_extreme_day}")
    print(f"Coldwave extreme day index: {cold_extreme_day}")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_event_day(true_t2m, predictions, best_thresholds, "Heatwave", heat_extreme_day)
    plot_event_day(true_t2m, predictions, best_thresholds, "Coldwave", cold_extreme_day)
    plot_enhanced_time_series(true_t2m, predictions, best_thresholds)
    plot_seasonal_event_analysis(true_t2m, predictions, best_thresholds)
    
    # Print optimal thresholds
    print("\nOptimized Thresholds:")
    for model, thresholds in best_thresholds.items():
        print(f"{model}:")
        print(f"  Heatwave threshold: {thresholds['Heatwave']:.4f}°C")
        print(f"  Coldwave threshold: {thresholds['Coldwave']:.4f}°C")
    
    print("All operations completed successfully")

if __name__ == "__main__":
    main()