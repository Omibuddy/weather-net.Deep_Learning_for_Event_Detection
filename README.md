# Heatwave Forecasting using Deep Learning

This project implements a deep learning approach to heatwave forecasting using a CNN-LSTM architecture with attention mechanisms.

## Project Structure
```
heatwave-forecasting/
├── data/                  # Data directory
├── docs/                  # Documentation
│   ├── figures/          # Generated figures
│   ├── tables/           # Generated tables
│   └── report.md         # Project report
├── models/               # Saved model checkpoints
├── results/              # Results directory
│   ├── raw/             # Raw results
│   └── processed/       # Processed results
├── visualizations/       # Generated visualizations
├── heatwave_training_pipeline.py  # Main training pipeline
├── load_data.py         # Data loading utilities
├── train_model.py       # Training script
├── visualize_results.py # Visualization script
└── requirements.txt     # Project dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**
   - Place your data files in the `data/` directory
   - Run data preprocessing if needed

2. **Training**
```bash
python train_model.py
```

3. **Visualization**
```bash
python visualize_results.py
```

## Model Architecture

The model combines:
- CNN layers for spatial feature extraction
- LSTM layers for temporal dependencies
- Multi-head attention for feature importance
- Residual connections for improved gradient flow

## Results

The model achieves:
- R² Score: 0.8535
- MSE: 0.1420
- MAE: 0.2921

Detailed results and visualizations are available in the `docs/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- List any acknowledgments here
- Cite relevant papers and resources 