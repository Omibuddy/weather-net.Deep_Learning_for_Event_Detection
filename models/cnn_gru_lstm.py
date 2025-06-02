import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# import tsaug # For data augmentation suggestions [1, 2, 3, 4]
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau # For LR scheduler suggestions [5, 6, 7, 8]

class WeatherAwarePositionalEncoding(nn.Module):
    """Seasonal and cyclical positional encoding for weather data with optional learnable components."""
    def __init__(self, d_model, max_len=365, learnable_pe=False):
        super().__init__()
        self.d_model = d_model
        self.learnable_pe = learnable_pe
        
        # Create positional encoding matrix for fixed components
        # max_len is set to 365 to cover a full year, allowing for flexible sequence lengths up to this max. [9]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Standard sinusoidal positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add seasonal components (if d_model allows)
        if d_model >= 4:
            # Annual cycle (365 days)
            pe[:, -4] = torch.sin(2 * math.pi * position.squeeze() / 365.25)
            pe[:, -3] = torch.cos(2 * math.pi * position.squeeze() / 365.25)
            # Monthly cycle (30 days)
            pe[:, -2] = torch.sin(2 * math.pi * position.squeeze() / 30.0)
            pe[:, -1] = torch.cos(2 * math.pi * position.squeeze() / 30.0)
        
        self.register_buffer('fixed_pe', pe.unsqueeze(0))

        # Learnable positional encoding [9]
        if self.learnable_pe:
            self.learnable_pe_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.xavier_uniform_(self.learnable_pe_embedding) # Initialize learnable PE

    def forward(self, x, day_of_year=None):
        seq_len = x.size(1)
        
        # Fixed positional encoding slice based on current sequence length
        fixed_pe_slice = self.fixed_pe[:, :seq_len]
        
        # Combine with learnable PE if enabled [9]
        if self.learnable_pe:
            learnable_pe_slice = self.learnable_pe_embedding[:, :seq_len]
            return x + fixed_pe_slice + learnable_pe_slice
        
        return x + fixed_pe_slice

class MultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution for different weather patterns with dilated convolutions."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different scales for weather patterns, incorporating dilation [10]
        # Dilation allows larger receptive fields without increasing parameters or losing resolution.
        self.conv_1h = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)  # Immediate
        self.conv_3h = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)  # Short-term
        self.conv_6h = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=2, dilation=2)  # Medium-term with dilation
        self.conv_12h = nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=4, dilation=2) # Long-term with dilation
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Apply different temporal scales
        conv1 = self.conv_1h(x)
        conv3 = self.conv_3h(x)
        conv6 = self.conv_6h(x)
        conv12 = self.conv_12h(x)
        
        # Concatenate multi-scale features
        combined = torch.cat([conv1, conv3, conv6, conv12], dim=1)
        combined = self.bn(combined)
        combined = F.relu(combined)
        combined = self.dropout(combined)
        
        return combined

class AdaptiveTemporalPooling(nn.Module):
    """Adaptive pooling that focuses on important temporal moments"""
    def __init__(self, input_dim, attention_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        
        # Weighted average and max pooling
        weighted_avg = torch.sum(attention_weights * x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        
        return torch.cat([weighted_avg, max_pool], dim=1), attention_weights

class WeatherFeatureExtractor(nn.Module):
    """
    Specialized feature extraction for weather data.
    
    This module assumes a specific ordering and grouping of input features.
    If you perform advanced feature engineering (e.g., adding lag features, rolling statistics,
    Fourier components, or interaction features [11, 12, 13, 14, 15, 16])
    before feeding data to the model, you MUST update the `num_features` parameter in the model
    initialization and adjust the slicing logic within this `forward` method
    to match your new feature set and their intended groupings.
    
    For example, if `num_features` increases to 20 due to new features, you would need to
    re-evaluate how these 20 features are grouped and adjust the input dimensions of
    `nn.Linear` layers and their corresponding slicing.
    """
    def __init__(self, num_features, hidden_dim=128):
        super().__init__()
        
        # These input dimensions (4, 3, 3, 3) sum to 13.
        # If num_features changes, these linear layers and their input dimensions must be updated.
        # This is a simplified example assuming the first 13 features are structured this way.
        self.temp_features = nn.Linear(4, hidden_dim//4) # e.g., Temp_max, Temp_min, Temp_avg, Dew_point
        self.pressure_features = nn.Linear(3, hidden_dim//4) # e.g., Pressure_sea, Pressure_avg, Pressure_max
        self.atmospheric_features = nn.Linear(3, hidden_dim//4) # e.g., Humidity, Wind_speed, Visibility
        self.condition_features = nn.Linear(3, hidden_dim//4) # e.g., Cloud_cover, Precipitation, Snowfall
        
        # A general linear layer to handle any remaining or new features if the total num_features > 13
        # This is a placeholder; ideally, you'd define specific layers for new feature groups.
        self.additional_features_linear = None
        if num_features > 13:
            # Ensure the sum of hidden_dim//4 * 4 + (num_features - 13) matches hidden_dim
            # This assumes hidden_dim is a multiple of 4.
            self.additional_features_linear = nn.Linear(num_features - 13, hidden_dim - (hidden_dim//4)*4)
            self.feature_fusion_input_dim = hidden_dim
        else:
            self.feature_fusion_input_dim = (hidden_dim//4)*4 # Should be equal to hidden_dim if hidden_dim is divisible by 4

        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Ensure x has enough features for slicing.
        if x.shape[-1] < 13:
            raise ValueError(f"Input x has {x.shape[-1]} features, but WeatherFeatureExtractor expects at least 13 for default slicing.")

        temp_feat = F.relu(self.temp_features(x[..., :4]))
        pressure_feat = F.relu(self.pressure_features(x[..., 4:7]))
        atm_feat = F.relu(self.atmospheric_features(x[..., 7:10]))
        cond_feat = F.relu(self.condition_features(x[..., 10:13]))
        
        combined = torch.cat([temp_feat, pressure_feat, atm_feat, cond_feat], dim=-1)

        if self.additional_features_linear:
            additional_feat = F.relu(self.additional_features_linear(x[..., 13:]))
            combined = torch.cat([combined, additional_feat], dim=-1)
        
        # Apply fusion only if we have batch dimension
        if len(combined.shape) == 3:  # (batch, seq, features)
            batch_size, seq_len = combined.shape[:2]
            combined_flat = combined.view(-1, combined.shape[-1])
            fused = self.feature_fusion(combined_flat)
            fused = fused.view(batch_size, seq_len, -1)
        else:  # (batch, features)
            fused = self.feature_fusion(combined)
            
        return fused

class WeatherTransformerBlock(nn.Module):
    """
    Transformer block optimized for weather sequences with optional attention regularization.
    Can be extended to use FSatten/SOatten by replacing nn.MultiheadAttention. [17, 18, 19, 20]
    """
    def __init__(self, d_model, nhead=8, dropout=0.1, attention_l1_reg_weight=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout) # Renamed to avoid conflict with parameter
        self.attention_l1_reg_weight = attention_l1_reg_weight # For Attention Logic Regularization [21, 22]
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(x, x, x)
        
        # Apply L1 regularization to attention weights [21, 22]
        # This encourages sparsity in attention maps, focusing on fewer, more effective dependencies.
        attention_reg_loss = 0.0
        if self.attention_l1_reg_weight > 0:
            # Sum of absolute values of attention weights, scaled by the regularization weight
            attention_reg_loss = torch.norm(attn_weights, p=1) * self.attention_l1_reg_weight
            
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # Feed-forward with residual
        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout_layer(ff_out))
        
        return x, attn_weights, attention_reg_loss

class UltraWeatherModel(nn.Module):
    """Ultra-enhanced weather prediction model"""
    def __init__(self, seq_len=7, num_features=13, hidden_dim=256, 
                 num_transformer_layers=4, num_heads=8, dropout=0.2,
                 num_quantiles=1, learnable_pe=False, attention_l1_reg_weight=0.0):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles # For Quantile Regression [23, 24, 25, 26, 27]
        
        # Specialized weather feature extraction
        self.weather_extractor = WeatherFeatureExtractor(num_features, hidden_dim)
        
        # Positional encoding for temporal patterns with learnable option [9]
        # max_len for positional encoding should be sufficient for the longest possible sequence
        self.pos_encoding = WeatherAwarePositionalEncoding(hidden_dim, max_len=365, learnable_pe=learnable_pe)
        
        # Multi-scale temporal convolutions with dilation [10]
        self.temporal_conv = MultiScaleTemporalConv(hidden_dim, hidden_dim)
        
        # Transformer layers for long-range dependencies with attention regularization [21, 22]
        self.transformer_layers = nn.ModuleList()
        
        # Adaptive temporal pooling
        self.adaptive_pool = AdaptiveTemporalPooling(hidden_dim, hidden_dim//4)
        
        # Final prediction layers - outputting multiple quantiles [23, 24, 25, 26, 27]
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 from adaptive pooling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, self.num_quantiles) # Output for multiple quantiles
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
    
    def forward(self, x, day_of_year=None):
        # Ensure input data is float32 for consistency with PyTorch models
        # Your input is float64, so convert it during data loading or here.
        x = x.to(torch.float32)

        # Extract weather-specific features
        weather_features = self.weather_extractor(x)
        
        # Add positional encoding
        weather_features = self.pos_encoding(weather_features, day_of_year)
        
        # Multi-scale temporal convolution
        # Reshape for conv1d: (batch, features, seq_len)
        conv_input = weather_features.permute(0, 2, 1)
        temporal_features = self.temporal_conv(conv_input)
        # Reshape back: (batch, seq_len, features)
        temporal_features = temporal_features.permute(0, 2, 1)
        
        # Apply transformer layers
        transformer_out = temporal_features
        attention_maps = []
        attention_reg_losses = [] # Collect regularization losses from each transformer block
        for transformer in self.transformer_layers:
            transformer_out, attention, reg_loss = transformer(transformer_out)
            attention_maps.append(attention)
            attention_reg_losses.append(reg_loss)
        
        # Adaptive temporal pooling
        pooled_features, pool_attention = self.adaptive_pool(transformer_out)
        
        # Final prediction
        prediction = self.predictor(pooled_features)
        
        # Sum up attention regularization losses from all transformer layers
        total_attention_reg_loss = sum(attention_reg_losses)
        
        return prediction, {
            'attention_maps': attention_maps,
            'pool_attention': pool_attention,
            'attention_reg_loss': total_attention_reg_loss # Return total regularization loss
        }

class QuantileWeatherLossV2(nn.Module):
    """
    Advanced loss function for weather prediction using Quantile Regression.
    Penalizes under-predictions more heavily for higher quantiles.
    Includes extreme temperature and seasonal weighting.
    Now also integrates attention regularization loss.
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9], extreme_temp_threshold=35.0, cold_threshold=0.0,
                 extreme_weight=3.0, seasonal_weight=1.5):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        self.extreme_temp_threshold = extreme_temp_threshold
        self.cold_threshold = cold_threshold
        self.extreme_weight = extreme_weight
        self.seasonal_weight = seasonal_weight
        
    def forward(self, predictions, targets, day_of_year=None, attention_reg_loss=0.0):
        # Ensure quantiles are on the same device as predictions/targets
        self.quantiles = self.quantiles.to(predictions.device)

        # Quantile loss calculation [23, 24, 25, 26, 27]
        # predictions shape: (batch_size, num_quantiles)
        # targets shape: (batch_size, 1) or (batch_size,)
        
        # Expand targets to match prediction's quantile dimension
        if targets.dim() == 1:
            targets = targets.unsqueeze(1) # (batch_size, 1)
        
        # Calculate errors for each quantile
        errors = targets - predictions # (batch_size, num_quantiles)
        
        # Calculate quantile loss for each quantile
        quantile_loss = torch.where(errors >= 0, self.quantiles * errors, (self.quantiles - 1) * errors)
        
        # Average across quantiles for a base loss per sample
        base_loss = quantile_loss.mean(dim=-1) # (batch_size,)

        # Apply extreme temperature weighting (applied to the base loss for each sample)
        if targets.dim() > 1:
            main_target = targets[:, 0] # Use the first target if multiple are present
        else:
            main_target = targets

        extreme_hot_mask = (main_target > self.extreme_temp_threshold).float()
        extreme_cold_mask = (main_target < self.cold_threshold).float()
        extreme_mask = extreme_hot_mask + extreme_cold_mask
        
        # Seasonal weighting (summer months get higher weight for heat prediction)
        seasonal_weight_factor = torch.ones_like(main_target)
        if day_of_year is not None:
            # Higher weight for summer months (June-August: days 152-243)
            # Ensure day_of_year is on the same device
            day_of_year = day_of_year.to(predictions.device)
            summer_mask = ((day_of_year >= 152) & (day_of_year <= 243)).float()
            seasonal_weight_factor = 1 + summer_mask * (self.seasonal_weight - 1)
        
        # Combined weighting
        total_weight = 1 + extreme_mask * self.extreme_weight
        total_weight = total_weight * seasonal_weight_factor
        
        weighted_loss = base_loss * total_weight
        
        # Add attention regularization loss to the total loss [21, 22]
        final_loss = weighted_loss.mean() + attention_reg_loss
        
        return final_loss

def create_model_with_config(config):
    """Create model with configuration dictionary"""
    model = UltraWeatherModel(
        seq_len=config.get('seq_len', 7),
        num_features=config.get('num_features', 13),
        hidden_dim=config.get('hidden_dim', 256),
        num_transformer_layers=config.get('num_transformer_layers', 4),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.2),
        num_quantiles=len(config.get('quantiles', [0.5])), # Pass number of quantiles
        learnable_pe=config.get('learnable_pe', False), # New parameter for learnable PE [9]
        attention_l1_reg_weight=config.get('attention_l1_reg_weight', 0.0) # New parameter for attention regularization [21, 22]
    )
    return model

def get_model_summary(model, input_shape):
    """Get model summary with input shape"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"Input Shape: {input_shape}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Trainable Parameters: {count_parameters(model):,}")