"""
Hyperparameter Tuning Script for Bike Demand Forecasting

This script provides comprehensive hyperparameter optimization for bike sharing demand models.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
import itertools
import time
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any

from bike_forecast_pytorch import BikeDataProcessor, BikeForecasterTrainer, BikeForecaster
from utils import set_seed, get_device, time_series_split

plt.style.use('seaborn-v0_8')

class HyperparameterTuner:
    """Class for hyperparameter tuning of bike demand prediction models"""
    
    def __init__(self, output_dir="outputs/tuning", device='auto'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
    def prepare_data(self, data_path=None, n_samples=5000, random_seed=42):
        """Prepare data for hyperparameter tuning (smaller dataset for speed)"""
        print("Preparing data for hyperparameter tuning...")
        
        # Load data
        processor = BikeDataProcessor()
        
        if data_path and Path(data_path).exists():
            df = pd.read_csv(data_path)
            if 'datetime' not in df.columns:
                df['datetime'] = pd.date_range(start='2011-01-01', periods=len(df), freq='H')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Take a contiguous tail slice when downsampling so chronological
            # order is preserved (random df.sample would scramble the timeline).
            if len(df) > n_samples:
                df = df.sort_values('datetime').tail(n_samples).reset_index(drop=True)
                print(f"Using last {n_samples} contiguous rows from dataset for faster tuning")
        else:
            df = processor.create_sample_data(n_samples=n_samples)

        # Prepare features
        X, y, feature_names = processor.prepare_data(df)

        # Split data (chronological)
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
    
    def define_search_spaces(self):
        """Define hyperparameter search spaces"""
        
        # MLP hyperparameters
        mlp_params = {
            'hidden_sizes': [
                [128, 64],
                [256, 128, 64],
                [512, 256, 128],
                [512, 256, 128, 64],
                [256, 128, 64, 32]
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128],
            'epochs': [30, 50, 100]
        }
        
        # LSTM hyperparameters
        lstm_params = {
            'hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128],
            'epochs': [30, 50, 100]
        }
        
        return {
            'mlp': mlp_params,
            'lstm': lstm_params
        }
    
    def create_model_with_params(self, model_type, params):
        """Create model with specified parameters"""
        
        if model_type == 'mlp':
            class CustomMLP(nn.Module):
                def __init__(self, input_size, hidden_sizes, dropout_rate):
                    super().__init__()
                    layers = []
                    prev_size = input_size
                    
                    for hidden_size in hidden_sizes:
                        layers.extend([
                            nn.Linear(prev_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate)
                        ])
                        prev_size = hidden_size
                    
                    layers.append(nn.Linear(prev_size, 1))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            return CustomMLP(
                input_size=self.X_train.shape[1],
                hidden_sizes=params['hidden_sizes'],
                dropout_rate=params['dropout_rate']
            )
        
        elif model_type == 'lstm':
            class CustomLSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0,
                        batch_first=True
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size // 2, 1)
                    )
                
                def forward(self, x):
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # Add sequence dimension
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(lstm_out[:, -1, :])
                    return out
            
            return CustomLSTM(
                input_size=self.X_train.shape[1],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
    
    def train_and_evaluate_model(self, model_type, params, max_time=300):
        """Train model with given parameters and return validation score"""
        
        try:
            start_time = time.time()
            
            # Create model
            model = self.create_model_with_params(model_type, params)
            model.to(self.device)
            
            # Create custom trainer
            class CustomTrainer(BikeForecasterTrainer):
                def build_model(self, input_size):
                    self.model = model
                    return self.model
            
            # Determine model type for trainer
            trainer_model_type = 'lstm' if model_type == 'lstm' else 'mlp'
            trainer = CustomTrainer(model_type=trainer_model_type, device=self.device)
            trainer.model = model

            # Train model with timeout
            trainer.train(
                self.X_train, self.y_train, self.X_val, self.y_val,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                lr=params['learning_rate']
            )

            training_time = time.time() - start_time

            # Early termination if taking too long
            if training_time > max_time:
                print(f"Training terminated early (>{max_time}s)")
                return -np.inf, training_time

            # Make predictions
            y_pred = trainer.predict(self.X_val)

            # Calculate metrics
            r2 = r2_score(self.y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

            return r2, training_time

        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return -np.inf, 0

    def grid_search(self, model_type='mlp', max_combinations=50, random_search=True):
        """Perform grid search or random search for hyperparameters"""

        print(f"\nStarting hyperparameter tuning for {model_type.upper()} model")
        print("=" * 60)

        search_spaces = self.define_search_spaces()
        param_space = search_spaces[model_type]

        # Generate parameter combinations
        if random_search and max_combinations < np.prod([len(v) for v in param_space.values()]):
            print(f"Using random search with {max_combinations} combinations")
            param_combinations = self.random_parameter_combinations(param_space, max_combinations)
        else:
            print("Using grid search")
            param_combinations = list(ParameterGrid(param_space))
            if len(param_combinations) > max_combinations:
                param_combinations = param_combinations[:max_combinations]
                print(f"Limited to first {max_combinations} combinations")

        print(f"Testing {len(param_combinations)} parameter combinations")

        best_score = -np.inf
        best_params = None

        for i, params in enumerate(param_combinations):
            print(f"\nTesting combination {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")

            score, training_time = self.train_and_evaluate_model(model_type, params)

            result = {
                'model_type': model_type,
                'params': params,
                'r2_score': score,
                'training_time': training_time,
                'combination_id': i
            }
            self.results.append(result)

            print(f"R² Score: {score:.4f}, Training Time: {training_time:.2f}s")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"New best score: {score:.4f}")

        self.best_score = best_score
        self.best_params = best_params

        print(f"\nBest parameters found:")
        print(f"Score: {best_score:.4f}")
        print(f"Parameters: {best_params}")

        return best_params, best_score

    def random_parameter_combinations(self, param_space, n_combinations):
        """Generate random parameter combinations"""

        combinations = []
        for _ in range(n_combinations):
            combination = {}
            for param, values in param_space.items():
                # Check if values is a list of lists (like hidden_sizes)
                if values and isinstance(values[0], (list, tuple)):
                    # Randomly select one of the list options
                    combination[param] = values[np.random.randint(len(values))]
                elif values and isinstance(values[0], (int, float, str)):
                    # Randomly select from scalar values
                    combination[param] = values[np.random.randint(len(values))]
                else:
                    # Fallback for other types
                    combination[param] = np.random.choice(values)
            combinations.append(combination)

        return combinations

    def bayesian_optimization(self, model_type='mlp', n_iterations=20):
        """Simple Bayesian optimization (placeholder for more advanced implementation)"""

        print(f"\nStarting Bayesian optimization for {model_type.upper()} model")
        print("=" * 60)
        print("Note: This is a simplified version. For production, consider using libraries like Optuna or Hyperopt.")

        # For now, use adaptive grid search based on previous results
        search_spaces = self.define_search_spaces()
        param_space = search_spaces[model_type]

        # Initial random exploration
        initial_combinations = self.random_parameter_combinations(param_space, n_iterations // 2)

        best_score = -np.inf
        best_params = None

        for i, params in enumerate(initial_combinations):
            print(f"\nIteration {i+1}/{len(initial_combinations)} (Exploration)")
            score, training_time = self.train_and_evaluate_model(model_type, params)

            result = {
                'model_type': model_type,
                'params': params,
                'r2_score': score,
                'training_time': training_time,
                'iteration': i,
                'phase': 'exploration'
            }
            self.results.append(result)

            if score > best_score:
                best_score = score
                best_params = params.copy()

        # Exploitation phase - refine around best parameters
        exploitation_combinations = self.generate_refinement_combinations(best_params, param_space, n_iterations // 2)

        for i, params in enumerate(exploitation_combinations):
            print(f"\nIteration {i+1}/{len(exploitation_combinations)} (Exploitation)")
            score, training_time = self.train_and_evaluate_model(model_type, params)

            result = {
                'model_type': model_type,
                'params': params,
                'r2_score': score,
                'training_time': training_time,
                'iteration': len(initial_combinations) + i,
                'phase': 'exploitation'
            }
            self.results.append(result)

            if score > best_score:
                best_score = score
                best_params = params.copy()

        self.best_score = best_score
        self.best_params = best_params

        return best_params, best_score

    def generate_refinement_combinations(self, base_params, param_space, n_combinations):
        """Generate parameter combinations around the best found parameters"""

        combinations = []
        for _ in range(n_combinations):
            combination = base_params.copy()

            # Randomly modify 1-2 parameters
            params_to_modify = np.random.choice(list(param_space.keys()),
                                              size=min(np.random.randint(1, 3), len(param_space)),
                                              replace=False)

            for param in params_to_modify:
                values = param_space[param]

                # Handle list of lists (like hidden_sizes)
                if values and isinstance(values[0], (list, tuple)):
                    if param in base_params and base_params[param] in values:
                        current_idx = values.index(base_params[param])
                        # Pick from nearby values
                        start_idx = max(0, current_idx - 1)
                        end_idx = min(len(values), current_idx + 2)
                        combination[param] = values[np.random.randint(start_idx, end_idx)]
                    else:
                        combination[param] = values[np.random.randint(len(values))]

                # Handle scalar values
                elif values and isinstance(values[0], (int, float, str)):
                    if param in base_params and base_params[param] in values:
                        current_idx = values.index(base_params[param])
                        # Pick from nearby values
                        start_idx = max(0, current_idx - 1)
                        end_idx = min(len(values), current_idx + 2)
                        combination[param] = values[np.random.randint(start_idx, end_idx)]
                    else:
                        combination[param] = values[np.random.randint(len(values))]
                else:
                    # Fallback
                    combination[param] = values[np.random.randint(len(values))]

            combinations.append(combination)

        return combinations

    def analyze_results(self, save_results=True):
        """Analyze and visualize hyperparameter tuning results"""

        if not self.results:
            print("No results to analyze")
            return

        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING RESULTS ANALYSIS")
        print("="*60)

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)

        # Expand parameters into separate columns
        param_columns = []
        for result in self.results:
            param_columns.extend(result['params'].keys())
        param_columns = list(set(param_columns))

        for param in param_columns:
            results_df[param] = results_df['params'].apply(lambda x: x.get(param, None))

        # Sort by score
        results_df = results_df.sort_values('r2_score', ascending=False)

        print(f"Total combinations tested: {len(results_df)}")
        print(f"Best R² score: {results_df['r2_score'].max():.4f}")
        print(f"Best parameters: {self.best_params}")

        # Top 10 results
        print("\nTop 10 Results:")
        top_results = results_df.head(10)[['r2_score', 'training_time'] + param_columns]
        print(top_results.round(4))

        if save_results:
            # Save results
            results_df.to_csv(self.output_dir / 'hyperparameter_tuning_results.csv', index=False)

            # Save best parameters
            with open(self.output_dir / 'best_parameters.json', 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_score': self.best_score,
                    'model_type': results_df.iloc[0]['model_type']
                }, f, indent=2)

        return results_df

    def visualize_results(self, results_df, save_plots=True):
        """Create visualizations of hyperparameter tuning results"""

        if results_df.empty:
            return

        print("\nGenerating hyperparameter tuning visualizations...")

        # Parameter impact analysis
        param_columns = [col for col in results_df.columns if col not in
                        ['model_type', 'params', 'r2_score', 'training_time', 'combination_id', 'iteration', 'phase']]

        n_params = len(param_columns)
        if n_params == 0:
            return

        # Create subplots for parameter analysis
        fig, axes = plt.subplots(2, min(3, (n_params + 1) // 2), figsize=(15, 10))
        if n_params == 1:
            axes = [axes]
        elif n_params <= 3:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        plot_idx = 0
        for i, param in enumerate(param_columns[:6]):  # Limit to 6 parameters
            if i >= len(axes):
                break

            ax = axes[i] if n_params > 1 else axes

            # Skip list-type parameters (like hidden_sizes)
            if results_df[param].dtype == 'object':
                # Check if it's a list
                first_value = results_df[param].iloc[0]
                if isinstance(first_value, (list, tuple)):
                    # Convert to string for plotting
                    results_df[f'{param}_str'] = results_df[param].apply(str)

                    # Group by string representation
                    param_str_col = f'{param}_str'
                    param_groups = results_df.groupby(param_str_col)['r2_score'].mean().sort_values(ascending=False)

                    # Plot top 5 configurations
                    top_configs = param_groups.head(5)
                    ax.barh(range(len(top_configs)), top_configs.values)
                    ax.set_yticks(range(len(top_configs)))
                    ax.set_yticklabels([label[:30] + '...' if len(label) > 30 else label
                                       for label in top_configs.index], fontsize=8)
                    ax.set_xlabel('Mean R² Score')
                    ax.set_title(f'{param} (Top 5)')
                    plot_idx += 1
                    continue

            # Numerical or categorical parameter
            try:
                if results_df[param].dtype in ['object', 'string'] or results_df[param].nunique() < 10:
                    # Categorical parameter - use box plot
                    param_groups = results_df.groupby(param)['r2_score'].apply(list)
                    ax.boxplot(param_groups.values, labels=param_groups.index)
                    ax.set_xticklabels(param_groups.index, rotation=45)
                    ax.set_ylabel('R² Score')
                    ax.set_title(f'Impact of {param}')
                else:
                    # Numerical parameter - use scatter plot
                    ax.scatter(results_df[param], results_df['r2_score'], alpha=0.6)
                    ax.set_xlabel(param)
                    ax.set_ylabel('R² Score')
                    ax.set_title(f'Impact of {param}')

                ax.grid(True, alpha=0.3)
                plot_idx += 1
            except Exception as e:
                print(f"Skipping visualization for {param}: {e}")
                continue

        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_plots:
            plt.savefig(self.output_dir / 'parameter_impact_analysis.png',
                       dpi=300, bbox_inches='tight')
        plt.show()

        # Performance vs training time
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(results_df['training_time'], results_df['r2_score'],
                            c=range(len(results_df)), cmap='viridis', alpha=0.6, s=50)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('R² Score')
        plt.title('Performance vs Training Time Trade-off')
        plt.colorbar(scatter, label='Experiment Order')
        plt.grid(True, alpha=0.3)

        if save_plots:
            plt.savefig(self.output_dir / 'performance_vs_time.png',
                       dpi=300, bbox_inches='tight')
        plt.show()

        # Convergence plot (if using iterative method)
        if 'iteration' in results_df.columns:
            plt.figure(figsize=(10, 6))

            # Plot best score over iterations
            best_scores = []
            current_best = -np.inf

            for _, row in results_df.iterrows():
                if row['r2_score'] > current_best:
                    current_best = row['r2_score']
                best_scores.append(current_best)

            plt.plot(range(1, len(best_scores) + 1), best_scores, 'b-', linewidth=2, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Best R² Score')
            plt.title('Hyperparameter Optimization Convergence')
            plt.grid(True, alpha=0.3)

            if save_plots:
                plt.savefig(self.output_dir / 'optimization_convergence.png',
                           dpi=300, bbox_inches='tight')
            plt.show()

    def validate_best_model(self, retrain=True):
        """Validate the best model on the test set"""

        if not self.best_params:
            print("No best parameters found. Run tuning first.")
            return None, None

        print("\n" + "="*60)
        print("VALIDATING BEST MODEL ON TEST SET")
        print("="*60)

        if retrain:
            print("Retraining best model on full training data...")

            # Combine train and validation sets for final training
            X_combined = np.vstack([self.X_train, self.X_val])
            y_combined = np.hstack([self.y_train, self.y_val])

            # Get model type from results
            model_type = self.results[0]['model_type'] if self.results else 'mlp'

            # Create and train model with best parameters
            model = self.create_model_with_params(model_type, self.best_params)
            model.to(self.device)

            # Train model
            class CustomTrainer(BikeForecasterTrainer):
                def build_model(self, input_size):
                    self.model = model
                    return self.model

            trainer_model_type = 'lstm' if model_type == 'lstm' else 'mlp'
            trainer = CustomTrainer(model_type=trainer_model_type, device=self.device)
            trainer.model = model

            trainer.train(
                X_combined, y_combined, self.X_test, self.y_test,  # Use test as validation for monitoring
                epochs=self.best_params['epochs'],
                batch_size=self.best_params['batch_size'],
                lr=self.best_params['learning_rate']
            )

            # Final predictions
            y_pred = trainer.predict(self.X_test)
        else:
            # Use the model from the best run (if available)
            print("Using model from best hyperparameter combination...")
            return None, None

        # Calculate final metrics
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        print(f"Final Test Performance:")
        print(f"R² Score: {test_r2:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"Best Parameters: {self.best_params}")

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        # Save final results
        final_results = {
            'best_params': convert_to_native(self.best_params),
            'validation_score': float(self.best_score),
            'test_score': float(test_r2),
            'test_rmse': float(test_rmse)
        }
        
        with open(self.output_dir / 'final_validation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return trainer, final_results
    
    def run_complete_tuning(self, data_path=None, model_type='mlp', method='grid', 
                          max_combinations=30, n_iterations=20, save_results=True):
        """Run the complete hyperparameter tuning pipeline"""
        
        print("Starting Comprehensive Hyperparameter Tuning")
        print("=" * 60)
        print(f"Model: {model_type}")
        print(f"Method: {method}")
        
        # Prepare data
        df = self.prepare_data(data_path)
        
        # Run hyperparameter optimization
        if method == 'grid':
            best_params, best_score = self.grid_search(model_type, max_combinations)
        elif method == 'random':
            best_params, best_score = self.grid_search(model_type, max_combinations, random_search=True)
        elif method == 'bayesian':
            best_params, best_score = self.bayesian_optimization(model_type, n_iterations)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Analyze results
        results_df = self.analyze_results(save_results)
        
        # Visualize results
        if save_results and results_df is not None:
            self.visualize_results(results_df, save_plots=True)
        
        # Validate best model
        trainer, final_results = self.validate_best_model(retrain=True)
        
        print(f"\n{'='*60}")
        print("HYPERPARAMETER TUNING COMPLETED!")
        print(f"Best validation score: {best_score:.4f}")
        if final_results:
            print(f"Final test score: {final_results['test_score']:.4f}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")
        
        return best_params, best_score, trainer, final_results

def main():
    """Main function for running hyperparameter tuning"""
    parser = argparse.ArgumentParser(description='Bike Demand Hyperparameter Tuning')
    parser.add_argument('--data', type=str, help='Path to bike sharing dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/tuning', 
                        help='Output directory for results')
    parser.add_argument('--model', type=str, choices=['mlp', 'lstm'], default='mlp',
                        help='Model type to tune')
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'bayesian'], 
                        default='random', help='Optimization method')
    parser.add_argument('--max-combinations', type=int, default=30, 
                        help='Maximum parameter combinations for grid/random search')
    parser.add_argument('--iterations', type=int, default=20, 
                        help='Number of iterations for Bayesian optimization')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], 
                        default='auto', help='Device for training')
    parser.add_argument('--samples', type=int, default=5000, 
                        help='Number of samples for tuning (smaller for speed)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create tuner
    tuner = HyperparameterTuner(output_dir=args.output_dir, device=args.device)
    
    # Run tuning
    best_params, best_score, trainer, final_results = tuner.run_complete_tuning(
        data_path=args.data,
        model_type=args.model,
        method=args.method,
        max_combinations=args.max_combinations,
        n_iterations=args.iterations
    )

if __name__ == "__main__":
    main()