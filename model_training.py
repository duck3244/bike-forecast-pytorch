"""
Model Training Script for Bike Demand Forecasting

This script provides focused model training functionality with advanced features.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
import argparse
from pathlib import Path
import json
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings('ignore')

from bike_forecast_pytorch import (
    BikeDataProcessor, BikeForecasterTrainer, 
    BikeForecaster, LSTMForecaster
)
from utils import (
    load_config, save_model, save_predictions,
    plot_learning_curves, plot_predictions_detailed,
    analyze_feature_importance_gradient, plot_feature_importance,
    create_model_report, set_seed, get_device, EarlyStopping
)

plt.style.use('seaborn-v0_8')

class AdvancedModelTrainer:
    """Advanced model training with comprehensive features"""
    
    def __init__(self, config_path: str = None, output_dir: str = "outputs/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = load_config(config_path)
        else:
            from utils import get_default_config
            self.config = get_default_config()
        
        # Setup device
        self.device = get_device()
        
        # Training history
        self.training_history = {}
        
    def prepare_data(self, data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        """Prepare data for training"""
        
        print("Preparing data for training...")
        
        processor = BikeDataProcessor()
        
        # Load or generate data
        if data_path and Path(data_path).exists():
            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            if 'datetime' not in df.columns:
                print("No datetime column found. Creating synthetic timestamps.")
                df['datetime'] = pd.date_range(start='2011-01-01', periods=len(df), freq='H')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            print("Generating synthetic data...")
            n_samples = self.config['data']['synthetic_samples']
            df = processor.create_sample_data(n_samples=n_samples)
        
        # Prepare features
        target_col = self.config['data']['target_column']
        X, y, feature_names = processor.prepare_data(df, target_col)
        
        # Verify data types
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

        # Additional verification
        if X.dtype != np.float32 and X.dtype != np.float64:
            print(f"Warning: Converting X from {X.dtype} to float32")
            X = X.astype(np.float32)
        if y.dtype != np.float32 and y.dtype != np.float64:
            print(f"Warning: Converting y from {y.dtype} to float32")
            y = y.astype(np.float32)

        return X, y, feature_names, df

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train/validation/test sets"""

        test_size = self.config['data']['test_size']
        val_size = self.config['data']['val_size']
        random_seed = self.config['data']['random_seed']

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )

        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_seed
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_model(self, model_type: str, input_size: int) -> nn.Module:
        """Create model based on configuration"""

        if model_type == 'mlp':
            config = self.config.get('models', {}).get('mlp', {})
            hidden_sizes = config.get('hidden_sizes', [512, 256, 128, 64])
            dropout_rate = config.get('dropout_rate', 0.2)

            return BikeForecaster(input_size, hidden_sizes, dropout_rate)

        elif model_type == 'lstm':
            config = self.config.get('models', {}).get('lstm', {})
            hidden_size = config.get('hidden_size', 128)
            num_layers = config.get('num_layers', 2)
            dropout = config.get('dropout', 0.2)

            return LSTMForecaster(input_size, hidden_size, num_layers, dropout)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str) -> BikeForecasterTrainer:
        """Train model with advanced features"""

        print(f"\nTraining {model_name} model...")
        print("-" * 40)

        # Ensure model is on correct device
        model = model.to(self.device)

        # Create trainer
        class CustomTrainer(BikeForecasterTrainer):
            def build_model(self, input_size):
                self.model = model
                self.model.to(self.device)
                return self.model

        # Determine model type
        model_type = 'lstm' if 'lstm' in model_name.lower() else 'mlp'

        trainer = CustomTrainer(model_type=model_type, device=self.device)
        trainer.model = model

        # Training parameters
        training_config = self.config['training']
        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        learning_rate = training_config['learning_rate']

        # Train with timing
        start_time = time.time()

        trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate
        )

        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")

        # Store training history
        self.training_history[model_name] = {
            'trainer': trainer,
            'training_time': training_time,
            'history': trainer.history
        }

        return trainer

    def evaluate_model(self, trainer: BikeForecasterTrainer, X_test: np.ndarray,
                      y_test: np.ndarray, model_name: str) -> Dict:
        """Comprehensive model evaluation"""

        print(f"\nEvaluating {model_name} model...")
        print("-" * 40)

        # Make predictions
        y_pred = trainer.predict(X_test)

        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }

        # Print results
        print(f"Test Performance:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        return metrics, y_pred

    def save_model_artifacts(self, trainer: BikeForecasterTrainer, model_name: str,
                           metrics: Dict, feature_names: List[str]) -> None:
        """Save model and related artifacts"""

        print(f"Saving {model_name} model artifacts...")

        # Create model-specific directory
        model_dir = self.output_dir / model_name.lower().replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / 'model.pth'
        metadata = {
            'model_name': model_name,
            'metrics': metrics,
            'feature_names': feature_names,
            'config': self.config,
            'training_time': self.training_history[model_name]['training_time']
        }
        save_model(trainer.model, str(model_path), metadata)

        # Save training history
        history_path = model_dir / 'training_history.json'
        history_data = {
            'train_loss': self.training_history[model_name]['history']['train_loss'],
            'val_loss': self.training_history[model_name]['history']['val_loss'],
            'training_time': self.training_history[model_name]['training_time']
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"Model artifacts saved to {model_dir}")

    def generate_visualizations(self, model_name: str, X_test: np.ndarray,
                              y_test: np.ndarray, y_pred: np.ndarray,
                              feature_names: List[str]) -> None:
        """Generate comprehensive visualizations"""

        print(f"Generating visualizations for {model_name}...")

    def generate_visualizations(self, model_name: str, X_test: np.ndarray,
                              y_test: np.ndarray, y_pred: np.ndarray,
                              feature_names: List[str]) -> None:
        """Generate comprehensive visualizations"""

        print(f"Generating visualizations for {model_name}...")

        # Create model-specific directories
        model_dir = self.output_dir / model_name.lower().replace(' ', '_')
        plots_dir = model_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Learning curves
        if model_name in self.training_history:
            history = self.training_history[model_name]['history']
            plot_path = plots_dir / 'learning_curves.png'
            plot_learning_curves(
                history['train_loss'],
                history['val_loss'],
                str(plot_path)
            )

        # Prediction plots
        plot_path = plots_dir / 'predictions.png'
        plot_predictions_detailed(
            y_test, y_pred,
            title=f"{model_name} Predictions",
            save_path=str(plot_path)
        )

        # Feature importance (for neural networks)
        try:
            trainer = self.training_history[model_name]['trainer']
            X_sample = torch.FloatTensor(X_test[:200]).to(self.device)

            importance_df = analyze_feature_importance_gradient(
                trainer.model, X_sample, feature_names
            )

            plot_path = plots_dir / 'feature_importance.png'
            plot_feature_importance(
                importance_df,
                title=f"{model_name} Feature Importance",
                save_path=str(plot_path)
            )

            # Save feature importance data
            importance_df.to_csv(plots_dir / 'feature_importance.csv', index=False)

        except Exception as e:
            print(f"Could not generate feature importance: {e}")

        print(f"Visualizations saved to {plots_dir}")

    def generate_model_report(self, model_name: str, metrics: Dict,
                            feature_names: List[str]) -> None:
        """Generate comprehensive model report"""

        print(f"Generating report for {model_name}...")

        # Create model-specific directory
        model_dir = self.output_dir / model_name.lower().replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)

        training_time = self.training_history[model_name]['training_time']

        # Load feature importance if available
        plots_dir = model_dir / 'plots'
        importance_path = plots_dir / 'feature_importance.csv'
        feature_importance = None
        if importance_path.exists():
            feature_importance = pd.read_csv(importance_path)

        # Generate report
        report = create_model_report(
            model_name, metrics, training_time, self.config, feature_importance
        )

        # Save report
        report_path = model_dir / f'{model_name.lower().replace(" ", "_")}_report.md'
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Report saved to {report_path}")

    def cross_validate_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                           k_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation"""

        print(f"\nPerforming {k_folds}-fold cross-validation for {model_type}...")
        print("-" * 50)

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.config['data']['random_seed'])

        cv_scores = []
        cv_times = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold + 1}/{k_folds}")

            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # Create and train model
            model = self.create_model(model_type, X.shape[1])

            start_time = time.time()
            trainer = self.train_model(model, X_train_cv, y_train_cv, X_val_cv, y_val_cv, f"{model_type}_cv_fold_{fold+1}")
            fold_time = time.time() - start_time

            # Evaluate
            y_pred_cv = trainer.predict(X_val_cv)
            fold_r2 = r2_score(y_val_cv, y_pred_cv)

            cv_scores.append(fold_r2)
            cv_times.append(fold_time)

            print(f"  Fold {fold + 1} R²: {fold_r2:.4f}")

        cv_results = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scores': cv_scores,
            'mean_time': np.mean(cv_times),
            'total_time': sum(cv_times)
        }

        print(f"\nCross-validation results:")
        print(f"  Mean R²: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"  Individual scores: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"  Total time: {cv_results['total_time']:.2f}s")

        return cv_results

    def ensemble_training(self, model_types: List[str], X_train: np.ndarray,
                         y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train ensemble of different models"""

        print(f"\nTraining ensemble of models: {model_types}")
        print("-" * 50)

        ensemble_models = {}
        ensemble_predictions = []

        for model_type in model_types:
            print(f"\nTraining {model_type} for ensemble...")

            # Create and train model
            model = self.create_model(model_type, X_train.shape[1])
            trainer = self.train_model(model, X_train, y_train, X_val, y_val, f"ensemble_{model_type}")

            # Get predictions
            y_pred = trainer.predict(X_test)
            ensemble_predictions.append(y_pred)
            ensemble_models[model_type] = trainer

        # Combine predictions (simple averaging)
        ensemble_pred = np.mean(ensemble_predictions, axis=0)

        # Evaluate ensemble
        ensemble_metrics = {
            'mse': mean_squared_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'r2': r2_score(y_test, ensemble_pred),
            'mape': np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        }

        print(f"\nEnsemble Performance:")
        print(f"  R² Score: {ensemble_metrics['r2']:.4f}")
        print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"  MAE: {ensemble_metrics['mae']:.4f}")

        # Save ensemble results
        ensemble_dir = self.output_dir / 'ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        save_predictions(y_test, ensemble_pred,
                        str(ensemble_dir / 'ensemble_predictions.json'),
                        {'model_types': model_types, 'method': 'averaging'})

        # Save individual model predictions for analysis
        for i, model_type in enumerate(model_types):
            save_predictions(y_test, ensemble_predictions[i],
                           str(ensemble_dir / f'{model_type}_predictions.json'))

        return {
            'ensemble_metrics': ensemble_metrics,
            'individual_predictions': ensemble_predictions,
            'ensemble_prediction': ensemble_pred,
            'models': ensemble_models
        }

    def learning_curve_analysis(self, model_type: str, X_train: np.ndarray,
                               y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                               training_sizes: List[float] = None) -> Dict:
        """Analyze learning curves with different training set sizes"""

        if training_sizes is None:
            training_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

        print(f"\nAnalyzing learning curves for {model_type}...")
        print(f"Training sizes: {training_sizes}")
        print("-" * 50)

        train_scores = []
        val_scores = []
        training_times = []

        for size in training_sizes:
            print(f"Training with {size*100:.0f}% of data...")

            # Sample training data
            n_samples = int(len(X_train) * size)
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]

            # Train model
            model = self.create_model(model_type, X_train.shape[1])

            start_time = time.time()
            trainer = self.train_model(model, X_train_subset, y_train_subset,
                                     X_val, y_val, f"{model_type}_size_{size}")
            training_time = time.time() - start_time

            # Evaluate on both sets
            train_pred = trainer.predict(X_train_subset)
            val_pred = trainer.predict(X_val)

            train_r2 = r2_score(y_train_subset, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            train_scores.append(train_r2)
            val_scores.append(val_r2)
            training_times.append(training_time)

            print(f"  Training R²: {train_r2:.4f}, Validation R²: {val_r2:.4f}")

        # Plot learning curve
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot([s*100 for s in training_sizes], train_scores, 'o-', label='Training Score')
        plt.plot([s*100 for s in training_sizes], val_scores, 'o-', label='Validation Score')
        plt.xlabel('Training Set Size (%)')
        plt.ylabel('R² Score')
        plt.title(f'{model_type} Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot([s*100 for s in training_sizes], training_times, 'o-', color='red')
        plt.xlabel('Training Set Size (%)')
        plt.ylabel('Training Time (seconds)')
        plt.title(f'{model_type} Training Time vs Data Size')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save learning curve plot - create directory structure
        model_dir = self.output_dir / model_type.lower()
        plots_dir = model_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(plots_dir / 'learning_curve_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'training_sizes': training_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'training_times': training_times
        }
    
    def run_comprehensive_training(self, data_path: Optional[str] = None, 
                                 model_types: List[str] = None,
                                 include_cross_validation: bool = False,
                                 include_ensemble: bool = False,
                                 include_learning_curves: bool = False) -> Dict:
        """Run comprehensive training pipeline"""
        
        if model_types is None:
            model_types = ['mlp']
        
        print("Starting Comprehensive Model Training Pipeline")
        print("=" * 60)
        print(f"Models to train: {model_types}")
        
        # Prepare data
        X, y, feature_names, df = self.prepare_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        results = {}
        
        # Train individual models
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"TRAINING {model_type.upper()} MODEL")
            print('='*60)
            
            # Create and train model
            model = self.create_model(model_type, X.shape[1])
            trainer = self.train_model(model, X_train, y_train, X_val, y_val, model_type)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(trainer, X_test, y_test, model_type)
            
            # Save artifacts
            self.save_model_artifacts(trainer, model_type, metrics, feature_names)
            
            # Generate visualizations
            self.generate_visualizations(model_type, X_test, y_test, y_pred, feature_names)
            
            # Generate report
            self.generate_model_report(model_type, metrics, feature_names)
            
            results[model_type] = {
                'metrics': metrics,
                'trainer': trainer,
                'predictions': y_pred
            }
            
            # Cross-validation
            if include_cross_validation:
                cv_results = self.cross_validate_model(model_type, X, y)
                results[model_type]['cross_validation'] = cv_results
            
            # Learning curve analysis
            if include_learning_curves:
                lc_results = self.learning_curve_analysis(model_type, X_train, y_train, X_val, y_val)
                results[model_type]['learning_curves'] = lc_results
        
        # Ensemble training
        if include_ensemble and len(model_types) > 1:
            print(f"\n{'='*60}")
            print("TRAINING ENSEMBLE MODEL")
            print('='*60)
            
            ensemble_results = self.ensemble_training(
                model_types, X_train, y_train, X_val, y_val, X_test, y_test
            )
            results['ensemble'] = ensemble_results
        
        # Generate summary
        self.generate_training_summary(results, model_types)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TRAINING COMPLETED!")
        print(f"Results saved to: {self.output_dir}")
        print('='*60)
        
        return results
    
    def generate_training_summary(self, results: Dict, model_types: List[str]) -> None:
        """Generate comprehensive training summary"""
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        summary_data = []
        
        for model_type in model_types:
            if model_type in results:
                metrics = results[model_type]['metrics']
                training_time = self.training_history[model_type]['training_time']
                
                summary_data.append({
                    'Model': model_type.upper(),
                    'R²': f"{metrics['r2']:.4f}",
                    'RMSE': f"{metrics['rmse']:.2f}",
                    'MAE': f"{metrics['mae']:.2f}",
                    'MAPE': f"{metrics['mape']:.2f}%",
                    'Time': f"{training_time:.2f}s"
                })
        
        # Add ensemble if available
        if 'ensemble' in results:
            ensemble_metrics = results['ensemble']['ensemble_metrics']
            summary_data.append({
                'Model': 'ENSEMBLE',
                'R²': f"{ensemble_metrics['r2']:.4f}",
                'RMSE': f"{ensemble_metrics['rmse']:.2f}",
                'MAE': f"{ensemble_metrics['mae']:.2f}",
                'MAPE': f"{ensemble_metrics['mape']:.2f}%",
                'Time': 'N/A'
            })
        
        # Create and display summary table
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(self.output_dir / 'training_summary.csv', index=False)
        
        # Determine best model
        if len(model_types) > 1:
            best_model = max(model_types, key=lambda x: results[x]['metrics']['r2'])
            best_r2 = results[best_model]['metrics']['r2']
            
            print(f"\n🏆 Best Model: {best_model.upper()} (R² = {best_r2:.4f})")
        
        print(f"\n📊 Summary saved to: {self.output_dir / 'training_summary.csv'}")

def main():
    """Main function for advanced model training"""
    parser = argparse.ArgumentParser(description='Advanced Bike Demand Model Training')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to bike sharing dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/training',
                        help='Output directory for results')
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['mlp', 'lstm'], default=['mlp'],
                        help='Model types to train')
    parser.add_argument('--cross-validation', action='store_true',
                        help='Include cross-validation')
    parser.add_argument('--ensemble', action='store_true',
                        help='Train ensemble model')
    parser.add_argument('--learning-curves', action='store_true',
                        help='Analyze learning curves')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create trainer
    trainer = AdvancedModelTrainer(
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Run training
    results = trainer.run_comprehensive_training(
        data_path=args.data,
        model_types=args.models,
        include_cross_validation=args.cross_validation,
        include_ensemble=args.ensemble,
        include_learning_curves=args.learning_curves
    )

if __name__ == "__main__":
    main()