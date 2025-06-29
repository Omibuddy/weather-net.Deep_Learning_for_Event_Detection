import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple, Dict, List

class BalancedClimateDataPreprocessor:
    """
    Balanced preprocessor for climate data - optimized for memory while preserving accuracy
    """
    
    def __init__(self, sequence_length: int = 10, forecast_horizon: int = 1, 
                 spatial_downsample: int = 2, use_float32: bool = True):
        """
        Initialize preprocessor with balanced optimization options
        
        Args:
            sequence_length: Number of time steps to use as input sequence
            forecast_horizon: Number of time steps to predict ahead
            spatial_downsample: Factor to downsample spatial dimensions (2 = half resolution)
            use_float32: Use float32 instead of float64 to halve memory usage
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.spatial_downsample = spatial_downsample
        self.use_float32 = use_float32
        self.scalers = {}
        self.data_stats = {}
        self.dtype = np.float32 if use_float32 else np.float64
        
    def load_netcdf_files(self, file_paths: List[str]) -> xr.Dataset:
        """Load and concatenate multiple NetCDF files"""
        datasets = []
        
        for file_path in file_paths:
            print(f"Loading {file_path}...")
            ds = xr.open_dataset(file_path)
            
            # Conservative spatial downsampling to preserve spatial patterns
            if self.spatial_downsample > 1:
                ds = ds.isel(latitude=slice(None, None, self.spatial_downsample),
                           longitude=slice(None, None, self.spatial_downsample))
            
            datasets.append(ds)
            
        combined_ds = xr.concat(datasets, dim='valid_time')
        combined_ds = combined_ds.sortby('valid_time')
        
        print(f"Combined dataset shape: {dict(combined_ds.dims)}")
        print(f"Variables: {list(combined_ds.data_vars)}")
        
        return combined_ds
    
    def preprocess_spatial_data(self, dataset: xr.Dataset) -> np.ndarray:
        """Preprocess spatial climate data with memory optimization"""
        variables = ['d2m', 't2m']
        
        data_arrays = []
        for var in variables:
            if var in dataset.data_vars:
                # Convert to desired dtype immediately
                data_arrays.append(dataset[var].values.astype(self.dtype))
            else:
                print(f"Warning: Variable {var} not found in dataset")
        
        # Stack along last dimension: (time, lat, lon, variables)
        data = np.stack(data_arrays, axis=-1)
        
        print(f"Preprocessed data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Memory usage: {data.nbytes / 1024**3:.2f} GB")
        
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    def create_large_patches(self, data: np.ndarray, patch_size: int = 64, 
                            overlap: int = 16) -> np.ndarray:
        """
        Create larger regional patches to preserve more spatial context
        
        Args:
            data: Input data (time, lat, lon, variables)
            patch_size: Size of each patch (patch_size x patch_size) - larger for better accuracy
            overlap: Overlap between patches - larger overlap for better continuity
            
        Returns:
            Patches array (n_patches, time, patch_size, patch_size, variables)
        """
        n_time, n_lat, n_lon, n_vars = data.shape
        
        # Check if patch size is larger than available spatial dimensions
        if patch_size > n_lat or patch_size > n_lon:
            print(f"Warning: Patch size {patch_size}x{patch_size} is larger than available spatial dimensions {n_lat}x{n_lon}")
            print("Using full spatial extent as a single patch")
            
            # Use the full spatial extent as a single patch
            patches = np.zeros((1, n_time, n_lat, n_lon, n_vars), dtype=self.dtype)
            patches[0] = data
            
            print(f"Created 1 patch using full spatial extent: {patches.shape}")
            print(f"Patches memory usage: {patches.nbytes / 1024**3:.2f} GB")
            
            return patches
        
        stride = patch_size - overlap
        
        # Calculate number of patches
        n_patches_lat = (n_lat - patch_size) // stride + 1
        n_patches_lon = (n_lon - patch_size) // stride + 1
        total_patches = n_patches_lat * n_patches_lon
        
        print(f"Creating {total_patches} patches of size {patch_size}x{patch_size} with {overlap} overlap")
        
        # Initialize patches array
        patches = np.zeros((total_patches, n_time, patch_size, patch_size, n_vars), 
                          dtype=self.dtype)
        
        patch_idx = 0
        for i in range(0, n_lat - patch_size + 1, stride):
            for j in range(0, n_lon - patch_size + 1, stride):
                patches[patch_idx] = data[:, i:i+patch_size, j:j+patch_size, :]
                patch_idx += 1
        
        print(f"Patches shape: {patches.shape}")
        print(f"Patches memory usage: {patches.nbytes / 1024**3:.2f} GB")
        
        return patches

    def smart_temporal_sampling(self, data: np.ndarray, sample_ratio: float = 0.7) -> np.ndarray:
        """
        Intelligently sample temporal data to reduce memory while preserving patterns
        
        Args:
            data: Input data (time, lat, lon, variables) or patches format
            sample_ratio: Ratio of time steps to keep (0.7 = keep 70% of time steps)
            
        Returns:
            Temporally sampled data
        """
        if len(data.shape) == 4:  # Full spatial data
            n_time = data.shape[0]
        else:  # Patches format
            n_time = data.shape[1]
            
        # Create smart sampling - keep every nth timestep but ensure even distribution
        n_keep = int(n_time * sample_ratio)
        indices = np.linspace(0, n_time - 1, n_keep, dtype=int)
        
        if len(data.shape) == 4:
            sampled_data = data[indices]
        else:
            sampled_data = data[:, indices]
            
        print(f"Temporal sampling: {n_time} -> {n_keep} timesteps ({sample_ratio:.1%})")
        return sampled_data

    def normalize_data(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Normalize the data using StandardScaler"""
        original_shape = data.shape
        
        # Check if data is empty
        if data.size == 0:
            raise ValueError("Cannot normalize empty data array. Check if patches were created correctly.")
        
        if len(original_shape) == 5:  # Patches format
            n_patches, n_time, n_lat, n_lon, n_vars = original_shape
            # Reshape to (samples, features) for scaling
            data_reshaped = data.reshape(-1, n_vars)
        else:  # Original format
            n_time, n_lat, n_lon, n_vars = original_shape
            data_reshaped = data.reshape(-1, n_vars)
        
        if fit_scaler:
            for i in range(n_vars):
                scaler = StandardScaler()
                data_reshaped[:, i] = scaler.fit_transform(
                    data_reshaped[:, i].reshape(-1, 1)).flatten()
                self.scalers[f'var_{i}'] = scaler
                
                self.data_stats[f'var_{i}_mean'] = scaler.mean_[0]
                self.data_stats[f'var_{i}_std'] = scaler.scale_[0]
        else:
            for i in range(n_vars):
                if f'var_{i}' in self.scalers:
                    data_reshaped[:, i] = self.scalers[f'var_{i}'].transform(
                        data_reshaped[:, i].reshape(-1, 1)).flatten()
        
        normalized_data = data_reshaped.reshape(original_shape)
        return normalized_data
    
    def create_sequences_from_patches(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from patches data
        
        Args:
            patches: Patches array (n_patches, time, patch_size, patch_size, variables)
            
        Returns:
            X: Input sequences (n_patches * n_sequences, sequence_length, patch_size, patch_size, variables)
            y: Target sequences (n_patches * n_sequences, forecast_horizon, patch_size, patch_size, variables)
        """
        n_patches, n_time, patch_size, patch_size_2, n_vars = patches.shape
        n_sequences_per_patch = n_time - self.sequence_length - self.forecast_horizon + 1
        
        if n_sequences_per_patch <= 0:
            raise ValueError(f"Not enough time steps for sequence creation.")
        
        total_sequences = n_patches * n_sequences_per_patch
        
        # Initialize arrays
        X = np.zeros((total_sequences, self.sequence_length, patch_size, patch_size_2, n_vars), 
                    dtype=self.dtype)
        y = np.zeros((total_sequences, self.forecast_horizon, patch_size, patch_size_2, n_vars), 
                    dtype=self.dtype)
        
        # Create sequences for each patch
        seq_idx = 0
        for patch_idx in range(n_patches):
            patch_data = patches[patch_idx]  # (time, patch_size, patch_size, variables)
            
            for i in range(n_sequences_per_patch):
                X[seq_idx] = patch_data[i:i + self.sequence_length]
                y[seq_idx] = patch_data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                seq_idx += 1
        
        print(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        print(f"Memory usage - X: {X.nbytes / 1024**3:.2f} GB, y: {y.nbytes / 1024**3:.2f} GB")
        
        return X, y

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences for time series prediction"""
        n_time, n_lat, n_lon, n_vars = data.shape
        n_sequences = n_time - self.sequence_length - self.forecast_horizon + 1
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough time steps for sequence creation.")
        
        X = np.zeros((n_sequences, self.sequence_length, n_lat, n_lon, n_vars), 
                    dtype=self.dtype)
        y = np.zeros((n_sequences, self.forecast_horizon, n_lat, n_lon, n_vars), 
                    dtype=self.dtype)
        
        for i in range(n_sequences):
            X[i] = data[i:i + self.sequence_length]
            y[i] = data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
        
        print(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        print(f"Memory usage - X: {X.nbytes / 1024**3:.2f} GB, y: {y.nbytes / 1024**3:.2f} GB")
        
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15, 
                   test_ratio: float = 0.15) -> Dict[str, np.ndarray]:
        """Split data into train/validation/test sets"""
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), random_state=42, shuffle=False
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size), random_state=42, shuffle=False
        )
        
        print(f"Data split:")
        print(f"  Training: X={X_train.shape}, y={y_train.shape}")
        print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
        print(f"  Testing: X={X_test.shape}, y={y_test.shape}")
        
        # Print memory usage
        total_memory = 0
        for name, data in [('X_train', X_train), ('y_train', y_train),
                          ('X_val', X_val), ('y_val', y_val),
                          ('X_test', X_test), ('y_test', y_test)]:
            memory_gb = data.nbytes / 1024**3
            total_memory += memory_gb
            print(f"  {name} memory: {memory_gb:.2f} GB")
        
        print(f"  Total memory: {total_memory:.2f} GB")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def save_processed_data_compressed(self, data_dict: Dict[str, np.ndarray], 
                                     output_dir: str = 'processed_data'):
        """Save processed data with compression"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data splits with compression
        for key, value in data_dict.items():
            np.savez_compressed(os.path.join(output_dir, f'{key}.npz'), data=value)
            print(f"Saved compressed {key} with shape {value.shape}")
        
        # Save scalers and metadata
        with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(self.scalers, f)
        
        metadata = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'spatial_downsample': self.spatial_downsample,
            'use_float32': self.use_float32,
            'data_stats': self.data_stats,
            'variables': ['d2m', 't2m']
        }
        
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved scalers and metadata to {output_dir}")

    def process_files_balanced(self, file_paths: List[str], output_dir: str = 'processed_data',
                              approach: str = 'moderate_patches', 
                              temporal_sample_ratio: float = 0.8):
        """
        Balanced preprocessing pipeline with multiple approaches
        
        Args:
            file_paths: List of NetCDF file paths
            output_dir: Directory to save processed data
            approach: 'full_spatial', 'moderate_patches', or 'conservative_patches'
            temporal_sample_ratio: Ratio of time steps to keep (0.8 = keep 80%)
        """
        print("Starting balanced climate data preprocessing...")
        
        # Load data with moderate spatial downsampling
        dataset = self.load_netcdf_files(file_paths)
        
        # Preprocess spatial data
        data = self.preprocess_spatial_data(dataset)
        
        if approach == 'full_spatial':
            print("Using full spatial approach with temporal sampling...")
            # Apply temporal sampling to reduce memory
            sampled_data = self.smart_temporal_sampling(data, temporal_sample_ratio)
            
            # Normalize data
            normalized_data = self.normalize_data(sampled_data, fit_scaler=True)
            
            # Create sequences
            X, y = self.create_sequences(normalized_data)
            
        elif approach == 'moderate_patches':
            print("Using moderate patches approach...")
            # Apply temporal sampling first
            sampled_data = self.smart_temporal_sampling(data, temporal_sample_ratio)
            
            # Create larger patches with more overlap
            patches = self.create_large_patches(sampled_data, patch_size=64, overlap=16)
            
            # Normalize patches
            normalized_patches = self.normalize_data(patches, fit_scaler=True)
            
            # Create sequences from patches
            X, y = self.create_sequences_from_patches(normalized_patches)
            
        elif approach == 'conservative_patches':
            print("Using conservative patches approach...")
            # Less aggressive temporal sampling
            sampled_data = self.smart_temporal_sampling(data, 0.9)
            
            # Larger patches with significant overlap
            patches = self.create_large_patches(sampled_data, patch_size=96, overlap=32)
            
            # Normalize patches
            normalized_patches = self.normalize_data(patches, fit_scaler=True)
            
            # Create sequences from patches
            X, y = self.create_sequences_from_patches(normalized_patches)
        
        # Split data
        data_splits = self.split_data(X, y)
        
        # Check if total memory is under 5GB
        total_memory = sum(arr.nbytes for arr in data_splits.values()) / 1024**3
        print(f"\nTotal dataset memory: {total_memory:.2f} GB")
        
        if total_memory > 5.0:
            print("WARNING: Dataset exceeds 5GB. Consider:")
            print("- Reducing temporal_sample_ratio")
            print("- Using smaller patch_size")
            print("- Increasing spatial_downsample")
        
        # Save with compression
        self.save_processed_data_compressed(data_splits, output_dir)
        
        print("Balanced preprocessing completed successfully!")
        
        return data_splits

    @staticmethod
    def load_compressed_data(output_dir: str) -> Dict[str, np.ndarray]:
        """Load compressed data"""
        data_dict = {}
        for split in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
            file_path = os.path.join(output_dir, f'{split}.npz')
            if os.path.exists(file_path):
                data_dict[split] = np.load(file_path)['data']
                print(f"Loaded {split} with shape {data_dict[split].shape}")
        return data_dict


# Usage examples with balanced approaches
if __name__ == "__main__":
    files = [
        "samples/data_stream-oper_stepType-instant.nc",
        "samples/data_stream-oper_stepType-instant1.nc", 
        "samples/data_stream-oper_stepType-instant2.nc"
    ]
    
    print("=== Approach 1: Conservative Patches (Best Accuracy) ===")
    # Target: ~3-4GB, High accuracy
    preprocessor_conservative = BalancedClimateDataPreprocessor(
        sequence_length=10,       # Keep original sequence length
        forecast_horizon=1,
        spatial_downsample=2,     # Moderate spatial downsampling
        use_float32=True         # Still use float32 for memory savings
    )
    
    data_splits = preprocessor_conservative.process_files_balanced(
        files, 
        output_dir='processed_climate_conservative',
        approach='conservative_patches',    # Large patches with significant overlap
        temporal_sample_ratio=0.85         # Keep 85% of time steps
    )
    
    print("\n=== Approach 2: Moderate Patches (Balanced) ===")
    # Target: ~2-3GB, Good accuracy
    preprocessor_moderate = BalancedClimateDataPreprocessor(
        sequence_length=8,        # Slightly shorter sequences
        forecast_horizon=1,
        spatial_downsample=2,
        use_float32=True
    )
    
    # Uncomment to run:
    # data_splits_mod = preprocessor_moderate.process_files_balanced(
    #     files, 
    #     output_dir='processed_climate_moderate',
    #     approach='moderate_patches',
    #     temporal_sample_ratio=0.75
    # )
    
    print("\n=== Approach 3: Full Spatial with Temporal Sampling ===")
    # Target: ~4-5GB, Best spatial resolution
    preprocessor_full = BalancedClimateDataPreprocessor(
        sequence_length=6,        # Shorter sequences to compensate
        forecast_horizon=1,
        spatial_downsample=2,     # Keep reasonable spatial resolution
        use_float32=True
    )
    
    # Uncomment to run:
    # data_splits_full = preprocessor_full.process_files_balanced(
    #     files, 
    #     output_dir='processed_climate_full_spatial',
    #     approach='full_spatial',
    #     temporal_sample_ratio=0.6  # More aggressive temporal sampling
    # )
    
    print("\n=== Loading compressed data example ===")
    loaded_data = BalancedClimateDataPreprocessor.load_compressed_data(
        'processed_climate_conservative'
    )
    
    # Load metadata
    with open('processed_climate_conservative/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print(f"Preprocessing parameters: {metadata}")