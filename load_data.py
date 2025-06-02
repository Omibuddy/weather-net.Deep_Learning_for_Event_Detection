import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

class DataLoadingError(Exception):
    """Custom exception for data loading errors"""
    pass

def validate_data(X, y, name):
    """Validate data shapes and values"""
    try:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise DataLoadingError(f"{name} data must be numpy arrays")
        
        if len(X.shape) != 3:
            raise DataLoadingError(f"{name} X must be 3D array (samples, time_steps, features)")
        
        if len(y.shape) != 1:
            raise DataLoadingError(f"{name} y must be 1D array (samples)")
        
        if X.shape[0] != y.shape[0]:
            raise DataLoadingError(f"{name} X and y must have same number of samples")
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise DataLoadingError(f"{name} data contains NaN values")
        
        if np.isinf(X).any() or np.isinf(y).any():
            raise DataLoadingError(f"{name} data contains infinite values")
            
    except Exception as e:
        raise DataLoadingError(f"Error validating {name} data: {str(e)}")

def load_data(batch_size=32):
    """
    Load and prepare data for training
    
    Args:
        batch_size (int): Batch size for DataLoader
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, input_shape)
    """
    try:
        print("Loading data...")
        
        # Check if data directory exists
        if not os.path.exists('data'):
            raise DataLoadingError("Data directory not found")
        
        # Check if all required files exist
        required_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
        for file in required_files:
            if not os.path.exists(os.path.join('data', file)):
                raise DataLoadingError(f"Required file {file} not found in data directory")
        
        # Load numpy arrays
        try:
            X_train = np.load('data/X_train.npy')
            y_train = np.load('data/y_train.npy')
            X_val = np.load('data/X_val.npy')
            y_val = np.load('data/y_val.npy')
            X_test = np.load('data/X_test.npy')
            y_test = np.load('data/y_test.npy')
        except Exception as e:
            raise DataLoadingError(f"Error loading numpy files: {str(e)}")
        
        # Validate data
        validate_data(X_train, y_train, "Training")
        validate_data(X_val, y_val, "Validation")
        validate_data(X_test, y_test, "Test")
        
        # Print data shapes
        print("\nData shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Convert to PyTorch tensors
        try:
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)
        except Exception as e:
            raise DataLoadingError(f"Error converting to PyTorch tensors: {str(e)}")
        
        # Create datasets
        try:
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
        except Exception as e:
            raise DataLoadingError(f"Error creating datasets: {str(e)}")
        
        # Create data loaders
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        except Exception as e:
            raise DataLoadingError(f"Error creating data loaders: {str(e)}")
        
        # Get input shape for model creation
        input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
        
        print("\nData loaded successfully!")
        print(f"Input shape: {input_shape}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, input_shape
        
    except DataLoadingError as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Test data loading
        train_loader, val_loader, test_loader, input_shape = load_data()
        
        # Print a sample batch
        for batch_x, batch_y in train_loader:
            print("\nSample batch:")
            print(f"X shape: {batch_x.shape}")
            print(f"y shape: {batch_y.shape}")
            print(f"X range: [{batch_x.min():.2f}, {batch_x.max():.2f}]")
            print(f"y range: [{batch_y.min():.2f}, {batch_y.max():.2f}]")
            break
    except Exception as e:
        print(f"Error in main: {str(e)}", file=sys.stderr)
        sys.exit(1) 