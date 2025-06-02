import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from load_data import load_data
from heatwave_training_pipeline import create_model_and_trainer
import os
from datetime import datetime

def plot_training_history(train_losses, val_losses, save_dir='visualizations'):
    """Plot training and validation loss history"""
    # Check if we have valid loss data
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("Warning: No training/validation loss data available. Skipping training history plot.")
        return
    
    # Convert losses to numpy arrays if they aren't already
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Raw losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7, linewidth=2)
    plt.title('Training and Validation Loss', fontsize=12, pad=15)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add min loss values to the plot
    min_train = np.min(train_losses)
    min_val = np.min(val_losses)
    plt.text(0.02, 0.98, f'Min Training Loss: {min_train:.4f}', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.02, 0.93, f'Min Validation Loss: {min_val:.4f}', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Log scale losses
    plt.subplot(1, 2, 2)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    plt.plot(np.log10(train_losses + epsilon), label='Log Training Loss', color='blue', alpha=0.7, linewidth=2)
    plt.plot(np.log10(val_losses + epsilon), label='Log Validation Loss', color='red', alpha=0.7, linewidth=2)
    plt.title('Log Scale Training and Validation Loss', fontsize=12, pad=15)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Log Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add min log loss values to the plot
    min_train_log = np.log10(min_train + epsilon)
    min_val_log = np.log10(min_val + epsilon)
    plt.text(0.02, 0.98, f'Min Log Training Loss: {min_train_log:.4f}', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(0.02, 0.93, f'Min Log Validation Loss: {min_val_log:.4f}', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, save_dir='visualizations'):
    """Plot predicted vs actual values"""
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.title(f'Predicted vs Actual Values\nMSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_vs_actual.png'))
    plt.close()

def plot_error_distribution(y_true, y_pred, save_dir='visualizations'):
    """Plot error distribution"""
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    errors = y_pred - y_true
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Error histogram using matplotlib instead of seaborn
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='blue', label='Error Distribution')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 2: Error vs Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, errors, alpha=0.5, s=1, color='blue', label='Errors')  # Reduced point size
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    plt.title('Error vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
    plt.close()

def plot_feature_importance(model, feature_names, save_dir='visualizations'):
    """Plot feature importance based on model attention"""
    # Create a hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # For MultiheadAttention, output[1] contains the attention weights
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights.append(output[1].detach().cpu().numpy())
    
    # Register the hook
    hook = model.attention.register_forward_hook(attention_hook)
    
    # Create a dummy input to get attention weights
    dummy_input = torch.randn(1, 7, 13).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)
    
    # Remove the hook
    hook.remove()
    
    if attention_weights:
        # Average attention weights across heads
        attention_weights = np.mean(attention_weights[0], axis=0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(attention_weights, 
                    xticklabels=feature_names,
                    yticklabels=range(1, attention_weights.shape[0] + 1),
                    cmap='YlOrRd')
        plt.title('Feature Importance (Attention Weights)')
        plt.xlabel('Features')
        plt.ylabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        plt.close()
    else:
        print("Warning: Could not capture attention weights. Skipping feature importance plot.")

def plot_time_series_predictions(y_true, y_pred, save_dir='visualizations'):
    """Plot time series predictions"""
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    plt.figure(figsize=(15, 6))
    
    # Plot actual values
    plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
    
    # Plot predictions
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    
    plt.title('Time Series Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_predictions.png'))
    plt.close()

def plot_prediction_intervals(y_true, y_pred, confidence=0.95, save_dir='visualizations'):
    """Plot predictions with confidence intervals"""
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate prediction intervals
    std = np.std(y_pred - y_true)
    z_score = 1.96  # for 95% confidence
    lower = y_pred - z_score * std
    upper = y_pred + z_score * std
    
    plt.figure(figsize=(15, 6))
    
    # Plot actual values
    plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
    
    # Plot predictions with confidence intervals
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    plt.fill_between(range(len(y_pred)), lower, upper, 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Predictions with Confidence Intervals')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_intervals.png'))
    plt.close()

def plot_monthly_performance(y_true, y_pred, save_dir='visualizations'):
    """Plot monthly performance metrics"""
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Assuming data is ordered chronologically and we have monthly data
    months = np.arange(len(y_true))
    monthly_mse = []
    monthly_mae = []
    monthly_r2 = []
    
    # Calculate metrics for each month
    for i in range(0, len(y_true), 30):  # Assuming 30 days per month
        end_idx = min(i + 30, len(y_true))
        monthly_mse.append(mean_squared_error(y_true[i:end_idx], y_pred[i:end_idx]))
        monthly_mae.append(mean_absolute_error(y_true[i:end_idx], y_pred[i:end_idx]))
        monthly_r2.append(r2_score(y_true[i:end_idx], y_pred[i:end_idx]))
    
    plt.figure(figsize=(15, 5))
    
    # Plot MSE
    plt.subplot(1, 3, 1)
    plt.plot(monthly_mse, 'b-', label='MSE')
    plt.title('Monthly MSE')
    plt.xlabel('Month')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(monthly_mae, 'r-', label='MAE')
    plt.title('Monthly MAE')
    plt.xlabel('Month')
    plt.ylabel('MAE')
    plt.grid(True, alpha=0.3)
    
    # Plot R²
    plt.subplot(1, 3, 3)
    plt.plot(monthly_r2, 'g-', label='R²')
    plt.title('Monthly R² Score')
    plt.xlabel('Month')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'monthly_performance.png'))
    plt.close()

def generate_all_visualizations(model, trainer, test_loader, feature_names):
    """Generate all visualizations for the presentation"""
    print("Starting visualization generation...")
    
    # Create visualization directory
    save_dir = 'visualizations'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created visualization directory: {save_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    print("Model moved to device")
    
    # Get predictions
    print("Starting prediction generation...")
    model.eval()
    predictions = []
    targets = []
    
    total_batches = len(test_loader)
    print(f"Total test batches to process: {total_batches}")
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if i % 10 == 0:  # Print progress every 10 batches
                print(f"Processing batch {i+1}/{total_batches}")
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch_y.numpy())
    
    print("Finished generating predictions")
    predictions = np.array(predictions)
    targets = np.array(targets)
    print(f"Prediction shape: {predictions.shape}")
    
    try:
        # Generate all plots
        print("Generating training history plot...")
        # Try different ways to access the loss history
        train_losses = []
        val_losses = []
        
        # Check if trainer has history attribute
        if hasattr(trainer, 'history'):
            train_losses = trainer.history.get('train_loss', [])
            val_losses = trainer.history.get('val_loss', [])
        # Check if trainer has loss attributes directly
        elif hasattr(trainer, 'train_losses') and hasattr(trainer, 'val_losses'):
            train_losses = trainer.train_losses
            val_losses = trainer.val_losses
        # Check if trainer has loss history in a different format
        elif hasattr(trainer, 'loss_history'):
            train_losses = trainer.loss_history.get('train', [])
            val_losses = trainer.loss_history.get('val', [])
        
        print(f"Training losses shape: {len(train_losses)}")
        print(f"Validation losses shape: {len(val_losses)}")
        
        if len(train_losses) > 0 and len(val_losses) > 0:
            plot_training_history(train_losses, val_losses, save_dir)
        else:
            print("Warning: No training/validation loss data found. Skipping training history plot.")
        
        print("Generating prediction vs actual plot...")
        plot_prediction_vs_actual(targets, predictions, save_dir)
        
        print("Generating error distribution plot...")
        plot_error_distribution(targets, predictions, save_dir)
        
        print("Generating feature importance plot...")
        plot_feature_importance(model, feature_names, save_dir)
        
        print("Generating time series predictions plot...")
        plot_time_series_predictions(targets, predictions, save_dir)
        
        print("Generating prediction intervals plot...")
        plot_prediction_intervals(targets, predictions, save_dir)
        
        print("Generating monthly performance plot...")
        plot_monthly_performance(targets, predictions, save_dir)
        
        print(f"All visualizations have been saved to the '{save_dir}' directory")
    except Exception as e:
        print(f"Error during visualization generation: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting visualization script...")
    
    print("Loading model and trainer...")
    # Load the trained model
    model, trainer = create_model_and_trainer(
        X_train=np.load('data/X_train.npy'),
        y_train=np.load('data/y_train.npy'),
        task_type='regression'
    )
    print("Model and trainer created")
    
    print("Loading model weights...")
    # Load the saved model weights
    model.load_state_dict(torch.load('models/heatwave_model.pth'))
    print("Model weights loaded")
    
    print("Loading test data...")
    # Load test data
    _, _, test_loader, _ = load_data()
    print("Test data loaded")
    
    # Feature names (update these based on your actual features)
    feature_names = [
        'Temperature', 'Humidity', 'Pressure', 'Wind Speed',
        'Wind Direction', 'Precipitation', 'Solar Radiation',
        'Cloud Cover', 'Visibility', 'Dew Point',
        'Heat Index', 'Wind Chill', 'UV Index'
    ]
    
    print("Starting visualization generation...")
    # Generate all visualizations
    generate_all_visualizations(model, trainer, test_loader, feature_names)
    print("Visualization script completed successfully!") 