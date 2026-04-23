"""
Model Comparison Script for Bike Demand Forecasting

This script compares different model architectures for bike sharing demand prediction.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import argparse
from pathlib import Path

from bike_forecast_pytorch import (
    BikeDataProcessor, BikeForecasterTrainer,
    BikeForecaster, LSTMForecaster, evaluate_model
)
from utils import set_seed, get_device, time_series_split

plt.style.use('seaborn-v0_8')

class ModelComparator:
    """Class for comparing different models for bike demand prediction"""
    
    def __init__(self, output_dir="outputs/comparison", device='auto'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        self.results = {}
        self.training_times = {}
        self.models = {}
        
    def prepare_data(self, data_path=None, n_samples=8760, random_seed=42):
        """Prepare data for model comparison"""
        print("Preparing data for model comparison...")
        
        # Load data
        processor = BikeDataProcessor()
        
        if data_path and Path(data_path).exists():
            df = pd.read_csv(data_path)
            if 'datetime' not in df.columns:
                df['datetime'] = pd.date_range(start='2011-01-01', periods=len(df), freq='H')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            df = processor.create_sample_data(n_samples=n_samples)
        
        # Prepare features
        X, y, feature_names = processor.prepare_data(df)
        
        # Split data (chronological; no shuffling for time series)
        X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
            X, y, test_size=0.15, val_size=0.15
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_names)}")
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.feature_names = feature_names
        
        return df
    
    def define_pytorch_models(self):
        """Define PyTorch model architectures"""
        
        class SimpleMLP(nn.Module):
            """Simple MLP with fewer parameters"""
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.network(x)

        class DeepMLP(nn.Module):
            """Deep MLP with more layers"""
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.network(x)

        class ResidualMLP(nn.Module):
            """MLP with residual connections"""
            def __init__(self, input_size):
                super().__init__()
                self.input_layer = nn.Linear(input_size, 256)
                
                self.block1 = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 256)
                )
                
                self.block2 = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128)
                )
                
                self.output_layer = nn.Linear(128, 1)
                self.downsample = nn.Linear(256, 128)
            
            def forward(self, x):
                x = torch.relu(self.input_layer(x))
                
                # Residual block 1
                residual = x
                x = self.block1(x)
                x = torch.relu(x + residual)
                
                # Residual block 2 with downsampling
                residual = self.downsample(x)
                x = self.block2(x)
                x = torch.relu(x + residual)
                
                return self.output_layer(x)
        
        return {
            'SimpleMLP': SimpleMLP,
            'StandardMLP': BikeForecaster,
            'DeepMLP': DeepMLP,
            'ResidualMLP': ResidualMLP,
            'LSTM': LSTMForecaster
        }
    
    def train_pytorch_model(self, model_class, model_name, epochs=50):
        """Train a PyTorch model and return predictions"""
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Create custom trainer class for different architectures
        class CustomTrainer(BikeForecasterTrainer):
            def build_model(self, input_size):
                if model_name == 'LSTM':
                    self.model = model_class(input_size)
                else:
                    self.model = model_class(input_size)
                self.model.to(self.device)
                return self.model
        
        trainer = CustomTrainer(device=self.device)
        model = trainer.build_model(input_size=self.X_train.shape[1])
        
        # Train with reduced epochs for comparison
        trainer.train(self.X_train, self.y_train, self.X_val, self.y_val, 
                      epochs=epochs, batch_size=64)
        
        # Make predictions
        y_pred = trainer.predict(self.X_test)
        
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        self.models[model_name] = trainer
        
        return y_pred
    
    def train_sklearn_model(self, model_class, model_name, **kwargs):
        """Train a scikit-learn model and return predictions"""
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        model = model_class(**kwargs)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        self.models[model_name] = model
        
        return y_pred
    
    def compare_all_models(self, epochs=30, include_sklearn=True):
        """Compare all models"""
        print("\n" + "="*60)
        print("STARTING MODEL COMPARISON")
        print("="*60)
        
        # Train PyTorch models
        pytorch_models = self.define_pytorch_models()
        
        for model_name, model_class in pytorch_models.items():
            try:
                y_pred = self.train_pytorch_model(model_class, model_name, epochs)
                metrics = {
                    'mse': mean_squared_error(self.y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'mae': mean_absolute_error(self.y_test, y_pred),
                    'r2': r2_score(self.y_test, y_pred)
                }
                self.results[model_name] = metrics
                print(f"{model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Train scikit-learn models for comparison
        if include_sklearn:
            sklearn_models = [
                (LinearRegression, "Linear Regression", {}),
                (Ridge, "Ridge Regression", {'alpha': 1.0, 'random_state': 42}),
                (Lasso, "Lasso Regression", {'alpha': 1.0, 'random_state': 42}),
                (RandomForestRegressor, "Random Forest", {'n_estimators': 100, 'random_state': 42}),
                (GradientBoostingRegressor, "Gradient Boosting", {'n_estimators': 100, 'random_state': 42}),
                (SVR, "Support Vector Regression", {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'})
            ]
            
            for model_class, model_name, kwargs in sklearn_models:
                try:
                    y_pred = self.train_sklearn_model(model_class, model_name, **kwargs)
                    metrics = {
                        'mse': mean_squared_error(self.y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        'mae': mean_absolute_error(self.y_test, y_pred),
                        'r2': r2_score(self.y_test, y_pred)
                    }
                    self.results[model_name] = metrics
                    print(f"{model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue
    
    def analyze_results(self):
        """Analyze and display comparison results"""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df['training_time'] = pd.Series(self.training_times)
        
        # Sort by R² score
        results_df = results_df.sort_values('r2', ascending=False)
        
        print("Model Performance Summary:")
        print("-" * 80)
        print(results_df.round(4))
        
        # Save results
        results_df.to_csv(self.output_dir / 'model_comparison_results.csv')
        
        return results_df
    
    def visualize_results(self, results_df, save_plots=True):
        """Create visualizations of model comparison results"""
        print("\nGenerating comparison visualizations...")
        
        # Performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # RMSE comparison
        axes[0, 0].barh(results_df.index, results_df['rmse'], color='lightcoral')
        axes[0, 0].set_title('Root Mean Squared Error (Lower is Better)')
        axes[0, 0].set_xlabel('RMSE')
        
        # R² comparison
        axes[0, 1].barh(results_df.index, results_df['r2'], color='lightblue')
        axes[0, 1].set_title('R² Score (Higher is Better)')
        axes[0, 1].set_xlabel('R²')
        
        # MAE comparison
        axes[1, 0].barh(results_df.index, results_df['mae'], color='lightgreen')
        axes[1, 0].set_title('Mean Absolute Error (Lower is Better)')
        axes[1, 0].set_xlabel('MAE')
        
        # Training time comparison
        axes[1, 1].barh(results_df.index, results_df['training_time'], color='gold')
        axes[1, 1].set_title('Training Time (seconds)')
        axes[1, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'model_comparison_metrics.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Performance vs Time trade-off analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs Training Time
        ax1.scatter(results_df['training_time'], results_df['r2'], s=100, alpha=0.7)
        for i, model in enumerate(results_df.index):
            ax1.annotate(model, (results_df.iloc[i]['training_time'], results_df.iloc[i]['r2']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax1.set_xlabel('Training Time (seconds)')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Performance vs Training Time Trade-off')
        ax1.grid(True)
        
        # RMSE vs Training Time
        ax2.scatter(results_df['training_time'], results_df['rmse'], s=100, alpha=0.7, color='red')
        for i, model in enumerate(results_df.index):
            ax2.annotate(model, (results_df.iloc[i]['training_time'], results_df.iloc[i]['rmse']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Error vs Training Time Trade-off')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'performance_vs_time.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_best_model(self, results_df, save_plots=True):
        """Analyze the best performing model in detail"""
        print("\n" + "="*60)
        print("BEST MODEL ANALYSIS")
        print("="*60)
        
        # Get best model
        best_model_name = results_df.index[0]
        best_model = self.models[best_model_name]
        
        print(f"Best performing model: {best_model_name}")
        print(f"R² Score: {results_df.loc[best_model_name, 'r2']:.4f}")
        print(f"RMSE: {results_df.loc[best_model_name, 'rmse']:.4f}")
        print(f"MAE: {results_df.loc[best_model_name, 'mae']:.4f}")
        print(f"Training Time: {results_df.loc[best_model_name, 'training_time']:.2f} seconds")
        
        # Make predictions with best model
        if hasattr(best_model, 'predict'):
            if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                                 'Random Forest', 'Gradient Boosting', 'Support Vector Regression']:
                y_pred_best = best_model.predict(self.X_test)
            else:
                y_pred_best = best_model.predict(self.X_test)
        else:
            y_pred_best = None
        
        if y_pred_best is not None:
            # Plot predictions vs actual for best model
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            ax1.scatter(self.y_test, y_pred_best, alpha=0.6, s=20)
            ax1.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predictions')
            ax1.set_title(f'{best_model_name} - Predictions vs Actual')
            ax1.grid(True)
            
            # Residual plot
            residuals = self.y_test - y_pred_best
            ax2.scatter(y_pred_best, residuals, alpha=0.6, s=20)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title(f'{best_model_name} - Residual Plot')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(self.output_dir / f'best_model_{best_model_name.lower().replace(" ", "_")}_analysis.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
        
        return best_model_name, best_model
    
    def analyze_feature_importance(self, results_df, save_plots=True):
        """Analyze feature importance for tree-based models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Find best tree-based model
        tree_models = ['Random Forest', 'Gradient Boosting']
        best_tree_model = None
        best_tree_score = -1
        
        for model_name in tree_models:
            if model_name in results_df.index:
                if results_df.loc[model_name, 'r2'] > best_tree_score:
                    best_tree_score = results_df.loc[model_name, 'r2']
                    best_tree_model = model_name
        
        if best_tree_model:
            model = self.models[best_tree_model]
            feature_importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(15), importance_df.head(15)['importance'])
            plt.yticks(range(15), importance_df.head(15)['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{best_tree_model} - Top 15 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(self.output_dir / f'feature_importance_{best_tree_model.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Top 10 Most Important Features ({best_tree_model}):")
            print(importance_df.head(10))
            
            # Save feature importance
            importance_df.to_csv(self.output_dir / f'feature_importance_{best_tree_model.lower().replace(" ", "_")}.csv', 
                               index=False)
            
            return importance_df
        
        else:
            print("No tree-based models available for feature importance analysis.")
            return None
    
    def generate_recommendations(self, results_df):
        """Generate model recommendations based on results"""
        print("\n" + "="*60)
        print("MODEL RECOMMENDATIONS")
        print("="*60)
        
        # Best overall performance
        best_r2_model = results_df.index[0]
        print(f"1. Best Overall Performance: {best_r2_model}")
        print(f"   R² Score: {results_df.loc[best_r2_model, 'r2']:.4f}")
        print(f"   RMSE: {results_df.loc[best_r2_model, 'rmse']:.4f}")
        
        # Fastest training
        fastest_model = results_df.loc[results_df['training_time'].idxmin()].name
        print(f"\n2. Fastest Training: {fastest_model}")
        print(f"   Training Time: {results_df.loc[fastest_model, 'training_time']:.2f} seconds")
        print(f"   R² Score: {results_df.loc[fastest_model, 'r2']:.4f}")
        
        # Best trade-off (high performance, reasonable training time)
        results_df['efficiency'] = results_df['r2'] / (results_df['training_time'] / 60)  # R² per minute
        most_efficient = results_df.loc[results_df['efficiency'].idxmax()].name
        print(f"\n3. Best Efficiency (Performance/Training Time): {most_efficient}")
        print(f"   Efficiency Score: {results_df.loc[most_efficient, 'efficiency']:.4f}")
        print(f"   R² Score: {results_df.loc[most_efficient, 'r2']:.4f}")
        print(f"   Training Time: {results_df.loc[most_efficient, 'training_time']:.2f} seconds")
        
        # Recommendations
        print("\n" + "="*60)
        print("DEPLOYMENT RECOMMENDATIONS:")
        print("\n• For Production Deployment:")
        if results_df.loc[fastest_model, 'r2'] > 0.85:
            print(f"  Use {fastest_model} for fast retraining and good performance")
        else:
            print(f"  Use {most_efficient} for balanced performance and speed")
        
        print("\n• For Maximum Accuracy:")
        print(f"  Use {best_r2_model} when prediction accuracy is most important")
        
        print("\n• For Research/Experimentation:")
        deep_learning_models = [m for m in results_df.index if any(x in m for x in ['MLP', 'LSTM'])]
        if deep_learning_models:
            best_dl = max(deep_learning_models, key=lambda x: results_df.loc[x, 'r2'])
            print(f"  Use {best_dl} as starting point for deep learning improvements")
        
        print("\n• Key Insights:")
        if any('MLP' in model for model in results_df.index[:3]):
            print("  - Neural networks show competitive performance for this problem")
        if any(model in results_df.index[:3] for model in ['Random Forest', 'Gradient Boosting']):
            print("  - Tree-based models remain strong baselines")
        if results_df.loc[results_df.index[0], 'r2'] - results_df.loc[results_df.index[-1], 'r2'] < 0.1:
            print("  - Performance differences are small - consider simplicity and speed")
        else:
            print("  - Clear performance differences exist - optimize for your specific needs")
        
        # Save recommendations
        with open(self.output_dir / 'model_recommendations.txt', 'w') as f:
            f.write("Model Comparison Recommendations\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Best Overall: {best_r2_model} (R²: {results_df.loc[best_r2_model, 'r2']:.4f})\n")
            f.write(f"Fastest: {fastest_model} ({results_df.loc[fastest_model, 'training_time']:.2f}s)\n")
            f.write(f"Most Efficient: {most_efficient} (Efficiency: {results_df.loc[most_efficient, 'efficiency']:.4f})\n")
    
    def run_complete_comparison(self, data_path=None, n_samples=8760, epochs=30, 
                               include_sklearn=True, save_plots=True):
        """Run the complete model comparison pipeline"""
        print("Starting Comprehensive Model Comparison")
        print("=" * 60)
        
        # Prepare data
        df = self.prepare_data(data_path, n_samples)
        
        # Compare all models
        self.compare_all_models(epochs, include_sklearn)
        
        # Analyze results
        results_df = self.analyze_results()
        
        # Visualize results
        if save_plots:
            self.visualize_results(results_df, save_plots)
        
        # Analyze best model
        best_model_name, best_model = self.analyze_best_model(results_df, save_plots)
        
        # Feature importance analysis
        if save_plots:
            self.analyze_feature_importance(results_df, save_plots)
        
        # Generate recommendations
        self.generate_recommendations(results_df)
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON COMPLETED!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")
        
        return results_df, best_model_name, best_model

def main():
    """Main function for running model comparison"""
    parser = argparse.ArgumentParser(description='Bike Demand Model Comparison')
    parser.add_argument('--data', type=str, help='Path to bike sharing dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison', 
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of epochs for PyTorch models')
    parser.add_argument('--no-sklearn', action='store_true', 
                        help='Skip scikit-learn models')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], 
                        default='auto', help='Device for PyTorch models')
    parser.add_argument('--samples', type=int, default=8760, 
                        help='Number of synthetic samples')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create comparator
    comparator = ModelComparator(output_dir=args.output_dir, device=args.device)
    
    # Run comparison
    results_df, best_model_name, best_model = comparator.run_complete_comparison(
        data_path=args.data,
        n_samples=args.samples,
        epochs=args.epochs,
        include_sklearn=not args.no_sklearn,
        save_plots=not args.no_plots
    )

if __name__ == "__main__":
    main()