import torch
from heatwave_training_pipeline import create_model_and_trainer
from load_data import load_data
import os
import sys
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class TrainingError(Exception):
    """Custom exception for training errors"""
    pass

def check_gpu():
    """Check GPU availability and memory"""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device)
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # Convert to GB
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
            return True
        else:
            logging.warning("No GPU available. Using CPU instead.")
            return False
    except Exception as e:
        logging.error(f"Error checking GPU: {str(e)}")
        return False

def train_model(batch_size=32, epochs=100, patience=15):
    """
    Train the heatwave prediction model
    
    Args:
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Check GPU availability
        has_gpu = check_gpu()
        
        # Load data
        logging.info("Loading data...")
        train_loader, val_loader, test_loader, input_shape = load_data(batch_size=batch_size)
        
        # Create model and trainer
        logging.info("Creating model and trainer...")
        try:
            model, trainer = create_model_and_trainer(
                X_train=train_loader.dataset.tensors[0],
                y_train=train_loader.dataset.tensors[1],
                task_type='regression'
            )
        except Exception as e:
            raise TrainingError(f"Error creating model: {str(e)}")
        
        # Train the model
        logging.info("Starting training...")
        try:
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                patience=patience,
                save_path='models/heatwave_model.pth'
            )
        except Exception as e:
            raise TrainingError(f"Error during training: {str(e)}")
        
        # Plot training history
        try:
            trainer.plot_training_history()
        except Exception as e:
            logging.warning(f"Error plotting training history: {str(e)}")
        
        # Save the full training state
        try:
            trainer.save_training_state('models/complete_training_state.pth')
        except Exception as e:
            logging.warning(f"Error saving full training state: {str(e)}")
        
        logging.info("Training completed successfully!")
        return model, trainer
        
    except TrainingError as e:
        logging.error(f"Training error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Train model
        model, trainer = train_model(
            batch_size=32,
            epochs=100,
            patience=15
        )
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1) 