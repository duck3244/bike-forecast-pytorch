#!/usr/bin/env python3
"""
PyTorch Bike Demand Forecasting - Main Execution Script
Enhanced version of EFavDB bike-forecast project

This script provides a complete pipeline for bike sharing demand prediction
using modern PyTorch implementations with comprehensive evaluation.
"""

import argparse
import time
from pathlib import Path
import torch
import pandas as pd
import numpy as np

from bike_forecast_pytorch import (
    BikeDataProcessor, BikeForecasterTrainer,
    BikeForecaster, LSTMForecaster, evaluate_model
)
from utils import (
    load_config, save_model, save_predictions,
    plot_learning_curves, plot_predictions_detailed,
    analyze_feature_importance_gradient, plot_feature_importance,
    create_model_report, create_data_summary, set_seed, get_device,
    time_series_split,
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PyTorch Bike Demand Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to bike sharing dataset (CSV format). If not provided, synthetic data will be generated.'
    )
    
    parser.add_argument(
        '--model', type=str, choices=['mlp', 'lstm'], default='mlp',
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Directory to save outputs'
    )
    
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Disable plot generation'
    )
    
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick training with reduced epochs'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
        help='Device to use for training'
    )
    
    return parser.parse_args()

def load_data(data_path, processor):
    """Load data from file or generate synthetic data"""
    if data_path and Path(data_path).exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            else:
                print("Warning: No datetime column found. Creating synthetic timestamps.")
                df['datetime'] = pd.date_range(start='2011-01-01', periods=len(df), freq='H')
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"Loaded {len(df)} samples from {data_path}")
        return df
    
    else:
        if data_path:
            print(f"Warning: Data file {data_path} not found. Generating synthetic data.")
        else:
            print("Generating synthetic data...")
        
        df = processor.create_sample_data(n_samples=8760)
        return df

def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == 'auto':
        device = get_device()
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def create_output_directories(output_dir):
    """Create output directories"""
    output_path = Path(output_dir)
    
    directories = [
        output_path,
        output_path / 'models',
        output_path / 'plots', 
        output_path / 'predictions',
        output_path / 'reports'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return output_path

def train_and_evaluate_model(config, df, model_type, device, output_path, generate_plots):
    """Complete training and evaluation pipeline"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Data preparation
    print("\nPreparing data...")
    processor = BikeDataProcessor()
    X, y, feature_names = processor.prepare_data(df, config['data']['target_column'])
    
    # Train-validation-test split (chronological; no shuffling for time series)
    test_size = config['data']['test_size']
    val_size = config['data']['val_size']

    X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
        X, y, test_size=test_size, val_size=val_size
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}")
    
    # Model initialization
    trainer = BikeForecasterTrainer(model_type=model_type, device=device)
    model = trainer.build_model(input_size=X.shape[1])
    
    # Training
    print(f"\nStarting training...")
    start_time = time.time()
    
    training_config = config['training']
    trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        lr=training_config['learning_rate']
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluation
    print("\nEvaluating model...")
    y_pred = trainer.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, f"{model_type.upper()} Model")
    
    # Save model
    model_path = output_path / 'models' / f'{model_type}_model.pth'
    metadata = {
        'model_type': model_type,
        'training_time': training_time,
        'metrics': metrics,
        'feature_names': feature_names,
        'config': config
    }
    save_model(trainer.model, str(model_path), metadata, processor=processor, feature_cols=feature_names)
    
    # Save predictions
    pred_path = output_path / 'predictions' / f'{model_type}_predictions.json'
    save_predictions(y_test, y_pred, str(pred_path), metadata)
    
    # Generate plots
    if generate_plots:
        print("Generating visualizations...")
        
        # Learning curves
        if hasattr(trainer, 'history'):
            plot_path = output_path / 'plots' / f'{model_type}_learning_curves.png'
            plot_learning_curves(
                trainer.history['train_loss'], 
                trainer.history['val_loss'],
                str(plot_path)
            )
        
        # Prediction plots
        plot_path = output_path / 'plots' / f'{model_type}_predictions.png'
        plot_predictions_detailed(
            y_test, y_pred, 
            title=f"{model_type.upper()} Model",
            save_path=str(plot_path)
        )
        
        # Feature importance (for neural networks)
        try:
            print("Calculating feature importance...")
            X_sample = torch.FloatTensor(X_test[:200]).to(device)
            importance_df = analyze_feature_importance_gradient(
                trainer.model, X_sample, feature_names
            )
            
            plot_path = output_path / 'plots' / f'{model_type}_feature_importance.png'
            plot_feature_importance(
                importance_df, 
                title=f"{model_type.upper()} Model Feature Importance",
                save_path=str(plot_path)
            )
            
        except Exception as e:
            print(f"Could not calculate feature importance: {e}")
            importance_df = None
    else:
        importance_df = None
    
    # Generate report
    print("Generating model report...")
    report = create_model_report(
        f"{model_type.upper()} Model", 
        metrics, 
        training_time, 
        config,
        importance_df
    )
    
    report_path = output_path / 'reports' / f'{model_type}_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Model report saved to {report_path}")
    
    return {
        'model': trainer,
        'metrics': metrics,
        'training_time': training_time,
        'predictions': (y_test, y_pred),
        'feature_importance': importance_df
    }

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("PyTorch Bike Demand Forecasting")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    try:
        config = load_config(args.config)
        if args.quick:
            config['training']['epochs'] = min(20, config['training']['epochs'])
            print("Quick mode: Reduced epochs to", config['training']['epochs'])
    except Exception as e:
        print(f"Error loading config: {e}")
        from utils import get_default_config
        config = get_default_config()
        print("Using default configuration")
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directories
    output_path = create_output_directories(args.output_dir)
    
    # Load data
    processor = BikeDataProcessor()
    df = load_data(args.data, processor)
    
    # Generate data summary
    print("\nGenerating data summary...")
    data_summary = create_data_summary(df)
    summary_path = output_path / 'reports' / 'data_summary.md'
    with open(summary_path, 'w') as f:
        f.write(data_summary)
    print(f"Data summary saved to {summary_path}")
    
    # Train and evaluate model
    results = train_and_evaluate_model(
        config, df, args.model, device, output_path, not args.no_plots
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Model Type: {args.model.upper()}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"R² Score: {results['metrics']['r2']:.4f}")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
    print(f"MAE: {results['metrics']['mae']:.4f}")
    print(f"\nOutputs saved to: {output_path}")
    print("\nFiles generated:")
    print(f"  - Model: models/{args.model}_model.pth")
    print(f"  - Predictions: predictions/{args.model}_predictions.json")
    print(f"  - Report: reports/{args.model}_report.md")
    print(f"  - Data Summary: reports/data_summary.md")
    if not args.no_plots:
        print(f"  - Plots: plots/{args.model}_*.png")
    
    print(f"\n{'='*60}")
    print("Bike demand forecasting completed successfully! 🚴‍♀️")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()