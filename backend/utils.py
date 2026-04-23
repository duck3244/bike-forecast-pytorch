# Utility functions for bike demand forecasting

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import json
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'data': {
            'synthetic_samples': 8760,
            'test_size': 0.2,
            'val_size': 0.2,
            'random_seed': 42,
            'target_column': 'count'
        },
        'training': {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'early_stopping_patience': 20
        },
        'models': {
            'mlp': {
                'hidden_sizes': [512, 256, 128, 64],
                'dropout_rate': 0.2
            }
        }
    }


def save_model(
    model: torch.nn.Module,
    path: str,
    metadata: Optional[Dict] = None,
    processor: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None,
) -> None:
    """Save model weights (.pth) plus a sidecar preprocessor.joblib.

    The weights file contains ONLY the state_dict — safe to load with
    ``weights_only=True``. The sidecar stores the fitted scaler, feature
    column order, and input size needed to reproduce the inference pipeline.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)
    logger.info(f"Model weights saved to {path}")

    sidecar = path.with_name('preprocessor.joblib')
    feature_names = (metadata or {}).get('feature_names') or feature_cols or []
    payload = {
        'model_class': model.__class__.__name__,
        'input_size': len(feature_names) if feature_names else None,
        'feature_cols': list(feature_names),
        'scalers': getattr(processor, 'scalers', {}) if processor is not None else {},
        'metadata': metadata or {},
    }
    joblib.dump(payload, sidecar)
    logger.info(f"Preprocessor sidecar saved to {sidecar}")


def load_checkpoint(model_dir: str, model_class_map: Dict[str, type]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model weights + preprocessor sidecar for inference.

    Expects ``model_dir`` to contain a ``*.pth`` weight file and a
    ``preprocessor.joblib`` sidecar written by :func:`save_model`. Weights
    are loaded with ``weights_only=True`` — the sidecar (pickle) must come
    from a trusted location.
    """
    import joblib

    model_dir = Path(model_dir)
    sidecar_path = model_dir / 'preprocessor.joblib'
    if not sidecar_path.exists():
        raise FileNotFoundError(f"preprocessor.joblib not found in {model_dir}")

    sidecar = joblib.load(sidecar_path)

    # Pick the weight file that matches the sidecar's model_type (e.g.
    # 'mlp_model.pth'). Falls back to any .pth if the expected name is
    # missing — keeps old checkpoints loadable.
    model_type = (sidecar.get('metadata') or {}).get('model_type')
    expected = model_dir / f'{model_type}_model.pth' if model_type else None
    if expected and expected.exists():
        weights_path = expected
    else:
        weight_files = sorted(model_dir.glob('*.pth'))
        if not weight_files:
            raise FileNotFoundError(f"No .pth weight file in {model_dir}")
        weights_path = weight_files[0]

    model_class_name = sidecar['model_class']
    if model_class_name not in model_class_map:
        raise ValueError(f"Unknown model class {model_class_name!r}; add it to model_class_map")

    model_cls = model_class_map[model_class_name]
    model = model_cls(input_size=sidecar['input_size'])
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded checkpoint from {model_dir}")
    return model, sidecar


def load_model(model_class: torch.nn.Module, path: str) -> Tuple[torch.nn.Module, Dict]:
    """Legacy loader kept for backward compatibility with old checkpoints."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    metadata = checkpoint.get('metadata', {})

    logger.info(f"Model loaded from {path}")
    return model, metadata


def save_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     path: str, metadata: Optional[Dict] = None) -> None:
    """Save predictions to file"""

    # Convert numpy types to native Python types
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
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(item) for item in obj)
        else:
            return obj

    results = {
        'y_true': y_true.tolist() if isinstance(y_true, np.ndarray) else y_true,
        'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'metadata': convert_to_native(metadata) if metadata else {},
        'metrics': convert_to_native(calculate_metrics(y_true, y_pred))
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Predictions saved to {path}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Avoid division by zero in MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


def plot_learning_curves(train_losses: List[float], val_losses: List[float],
                         save_path: Optional[str] = None) -> None:
    """Plot training and validation learning curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {save_path}")

    plt.show()


def plot_predictions_detailed(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Model Predictions",
                              save_path: Optional[str] = None) -> None:
    """Create detailed prediction plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title(f'{title} - Scatter Plot')
    axes[0, 0].grid(True, alpha=0.3)

    # Add R² to scatter plot
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Residual plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{title} - Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Time series plot (first 100 points)
    n_points = min(100, len(y_true))
    axes[1, 0].plot(range(n_points), y_true[:n_points], label='Actual', marker='o', markersize=4)
    axes[1, 0].plot(range(n_points), y_pred[:n_points], label='Predicted', marker='s', markersize=4)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].set_title(f'{title} - Time Series (First 100 Points)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Error distribution
    errors = np.abs(y_true - y_pred)
    axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'{title} - Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    # Add MAE to histogram
    mae = np.mean(errors)
    axes[1, 1].axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.2f}')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed prediction plots saved to {save_path}")

    plt.show()


def analyze_feature_importance_gradient(model: torch.nn.Module, X_sample: torch.Tensor,
                                        feature_names: List[str], n_samples: int = 100) -> pd.DataFrame:
    """Calculate feature importance using gradient-based method"""
    model.eval()

    # Select random sample
    if len(X_sample) > n_samples:
        indices = torch.randperm(len(X_sample))[:n_samples]
        X_sample = X_sample[indices]

    # Ensure tensor is on the same device as model and requires grad
    device = next(model.parameters()).device
    X_sample = X_sample.to(device)
    X_sample.requires_grad_(True)

    # Forward pass
    output = model(X_sample).sum()

    # Backward pass
    output.backward()

    # Calculate feature importance as mean absolute gradient
    # Move to CPU before converting to numpy
    importance = torch.abs(X_sample.grad).mean(dim=0).cpu().detach().numpy()

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                            title: str = "Feature Importance",
                            save_path: Optional[str] = None) -> None:
    """Plot feature importance"""
    plt.figure(figsize=(10, max(6, top_n * 0.4)))

    top_features = importance_df.head(top_n)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'{title} - Top {top_n} Features')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()


def create_model_report(model_name: str, metrics: Dict[str, float],
                        training_time: float, config: Dict[str, Any],
                        feature_importance: Optional[pd.DataFrame] = None) -> str:
    """Create a comprehensive model report"""

    # Convert config to YAML string safely
    config_yaml = yaml.dump(config.get('models', {}), default_flow_style=False)

    # Get MAPE value safely
    mape_value = metrics.get('mape', 'N/A')
    mape_str = f"{mape_value:.2f}%" if isinstance(mape_value, (int, float)) else str(mape_value)

    report = f"""
# Model Performance Report
## Model: {model_name}

### Performance Metrics
- **R² Score**: {metrics['r2']:.4f}
- **RMSE**: {metrics['rmse']:.4f}
- **MAE**: {metrics['mae']:.4f}
- **MSE**: {metrics['mse']:.4f}
- **MAPE**: {mape_str}

### Training Information
- **Training Time**: {training_time:.2f} seconds
- **Epochs**: {config.get('training', {}).get('epochs', 'N/A')}
- **Batch Size**: {config.get('training', {}).get('batch_size', 'N/A')}
- **Learning Rate**: {config.get('training', {}).get('learning_rate', 'N/A')}

### Model Configuration
```yaml
{config_yaml}```

### Performance Interpretation
- **Excellent**: R² > 0.9, RMSE < 10
- **Good**: R² > 0.8, RMSE < 20
- **Fair**: R² > 0.7, RMSE < 30
- **Poor**: R² < 0.7, RMSE > 30

**Current Performance**: {get_performance_level(metrics['r2'], metrics['rmse'])}

### Recommendations
{get_recommendations(metrics, training_time)}
    """

    if feature_importance is not None:
        report += f"""
### Top 10 Most Important Features
{feature_importance.head(10).to_string(index=False)}
        """

    return report


def get_performance_level(r2: float, rmse: float) -> str:
    """Get performance level based on metrics"""
    if r2 > 0.9 and rmse < 10:
        return "Excellent"
    elif r2 > 0.8 and rmse < 20:
        return "Good"
    elif r2 > 0.7 and rmse < 30:
        return "Fair"
    else:
        return "Poor"


def get_recommendations(metrics: Dict[str, float], training_time: float) -> str:
    """Generate recommendations based on performance"""
    recommendations = []

    if metrics['r2'] < 0.8:
        recommendations.append("- Consider increasing model complexity or adding more features")
        recommendations.append("- Check for data quality issues or missing important features")

    if metrics['rmse'] > 25:
        recommendations.append("- High RMSE indicates large prediction errors")
        recommendations.append("- Consider ensemble methods or regularization techniques")

    if training_time > 300:  # 5 minutes
        recommendations.append("- Long training time - consider model optimization")
        recommendations.append("- Use early stopping or reduce model complexity for faster training")

    if metrics['r2'] > 0.9:
        recommendations.append("- Excellent performance! Monitor for overfitting on new data")
        recommendations.append("- Consider this model for production deployment")

    if not recommendations:
        recommendations.append("- Performance is acceptable for most use cases")
        recommendations.append("- Continue monitoring model performance on new data")

    return "\n".join(recommendations)


def create_data_summary(df: pd.DataFrame) -> str:
    """Create a summary of the dataset"""

    summary = f"""
# Data Summary Report

## Dataset Overview
- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB
- **Date Range**: {df['datetime'].min()} to {df['datetime'].max()}

## Missing Values
{df.isnull().sum().to_string() if df.isnull().sum().sum() > 0 else "No missing values"}

## Numerical Features Summary
{df.describe().round(2).to_string()}

## Target Variable (count)
- **Mean**: {df['count'].mean():.2f}
- **Std**: {df['count'].std():.2f}
- **Min**: {df['count'].min()}
- **Max**: {df['count'].max()}
- **Skewness**: {df['count'].skew():.2f}
- **Kurtosis**: {df['count'].kurtosis():.2f}

## Data Quality Checks
- **Duplicate Rows**: {df.duplicated().sum()}
- **Zero Values in Count**: {(df['count'] == 0).sum()}
- **Negative Values in Count**: {(df['count'] < 0).sum()}
    """

    return summary


class EarlyStopping:
    """Early stopping utility class"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if training should be stopped"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    When deterministic is True, enables cuDNN deterministic mode so the same
    seed reproduces identical results across runs (at a small speed cost).
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time-ordered arrays into (train, val, test) without shuffling.

    Assumes X and y are already sorted chronologically. The tail of the
    sequence becomes the test set, the block before it the validation set,
    and the rest is training data. Using a chronological split prevents
    leakage from future observations into training.
    """
    n = len(X)
    if len(y) != n:
        raise ValueError(f"X and y length mismatch: {n} vs {len(y)}")

    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes: test={test_size}, val={val_size} "
            f"leaves {n_train} training samples"
        )

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device