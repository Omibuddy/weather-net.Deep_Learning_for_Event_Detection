import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from heatwave_training_pipeline import ImprovedCNNLSTMModel
from load_data import load_data
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model(model_path='models/complete_training_state.pth'):
    """Load the trained model"""
    # Load the complete training state
    checkpoint = torch.load(model_path)
    
    # Initialize model with saved parameters
    model = ImprovedCNNLSTMModel(
        seq_len=7,
        num_features=13,
        lstm_units=128,
        dense_units=64,
        num_classes=1,
        dropout_rate=0.2,
        task_type='regression'
    )
    
    # Load model state - using best_model_state which contains the best performing model
    model.load_state_dict(checkpoint['best_model_state'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    return model

def make_predictions(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Make predictions on the test set"""
    model = model.to(device)
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc="Making predictions"):
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    return np.array(predictions), np.array(actuals)

def plot_predictions(predictions, actuals, dates=None):
    """Plot predictions vs actual values"""
    # Flatten arrays if they're multi-dimensional
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    
    plt.figure(figsize=(15, 6))
    
    # Plot actual values
    plt.plot(actuals, label='Actual', color='blue', alpha=0.7)
    
    # Plot predictions
    plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
    
    # Add confidence intervals
    std = np.std(predictions - actuals)
    plt.fill_between(range(len(predictions)), 
                    predictions - 1.96*std, 
                    predictions + 1.96*std, 
                    color='red', alpha=0.2, 
                    label='95% Confidence Interval')
    
    plt.title('Heatwave Predictions vs Actual Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('docs/figures/predictions.png')
    plt.close()

def analyze_predictions(predictions, actuals):
    """Analyze prediction accuracy and errors"""
    print("Calculating metrics...")
    # Flatten arrays
    predictions = predictions.flatten()
    actuals = actuals.flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
    
    print("Creating analysis plots...")
    # Create analysis plots
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(actuals, predictions, alpha=0.5, s=1)  # Reduced point size
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Error vs Predicted
    plt.subplot(1, 2, 2)
    errors = predictions - actuals
    plt.scatter(predictions, errors, alpha=0.5, s=1)  # Reduced point size
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Error vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Error')
    
    print("Saving analysis plots...")
    plt.tight_layout()
    plt.savefig('docs/figures/prediction_analysis.png')
    plt.close()
    
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Mean Error': np.mean(errors),
        'Std Error': np.std(errors)
    }

def main():
    print("Loading model and data...")
    model = load_model()
    _, _, test_loader, _ = load_data(batch_size=32)
    
    print("Making predictions...")
    predictions, actuals = make_predictions(model, test_loader)
    print(f"Generated {len(predictions)} predictions")
    
    print("Plotting predictions...")
    plot_predictions(predictions, actuals)
    print("Predictions plot saved to docs/figures/predictions.png")
    
    print("Analyzing predictions...")
    metrics = analyze_predictions(predictions, actuals)
    
    print("\nPrediction Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nSaving predictions to CSV...")
    # Flatten arrays before creating DataFrame
    predictions_flat = predictions.flatten()
    actuals_flat = actuals.flatten()
    errors_flat = predictions_flat - actuals_flat
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'Actual': actuals_flat,
        'Predicted': predictions_flat,
        'Error': errors_flat
    })
    results_df.to_csv('results/processed/predictions.csv', index=False)
    print("Predictions saved to 'results/processed/predictions.csv'")

if __name__ == "__main__":
    main() 