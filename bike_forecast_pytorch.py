# PyTorch Bike Demand Forecasting
# Enhanced version of EFavDB bike-forecast project using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BikeDataset(Dataset):
    """Custom PyTorch Dataset for bike sharing data"""

    def __init__(self, features, targets):
        # Ensure features and targets are numpy arrays
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        # Convert to float32 explicitly
        features = features.astype(np.float32)
        targets = targets.astype(np.float32)

        # Convert to tensors
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class BikeForecaster(nn.Module):
    """Deep Neural Network for bike demand forecasting"""

    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.2):
        super(BikeForecaster, self).__init__()

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

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting"""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
        # LSTM output
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = self.fc(lstm_out[:, -1, :])
        return out


class BikeDataProcessor:
    """Data preprocessing and feature engineering for bike sharing data"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def create_sample_data(self, n_samples=10000):
        """Create synthetic bike sharing data for demonstration"""

        np.random.seed(42)

        # Time features
        dates = pd.date_range(start='2011-01-01', periods=n_samples, freq='H')

        data = {
            'datetime': dates,
            'season': np.random.choice([1, 2, 3, 4], n_samples),
            'holiday': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
            'workingday': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'weather': np.random.choice([1, 2, 3, 4], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'temp': np.random.normal(20, 10, n_samples),  # Temperature
            'atemp': np.random.normal(23, 12, n_samples),  # "Feels like" temperature
            'humidity': np.random.uniform(0, 100, n_samples),
            'windspeed': np.random.exponential(15, n_samples)
        }

        df = pd.DataFrame(data)

        # Create realistic bike count based on features
        base_count = 50

        # Season effect
        season_effect = {1: 0.8, 2: 1.2, 3: 1.3, 4: 1.0}
        df['season_mult'] = df['season'].map(season_effect)

        # Time of day effect
        df['hour'] = df['datetime'].dt.hour
        hour_effect = np.where(
            (df['hour'] >= 7) & (df['hour'] <= 9), 1.8,  # Morning rush
            np.where(
                (df['hour'] >= 17) & (df['hour'] <= 19), 1.6,  # Evening rush
                np.where(
                    (df['hour'] >= 10) & (df['hour'] <= 16), 1.2,  # Daytime
                    0.4  # Night/early morning
                )
            )
        )

        # Weather effect
        weather_effect = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.2}
        df['weather_mult'] = df['weather'].map(weather_effect)

        # Temperature effect (optimal around 25°C)
        temp_effect = 1 - np.abs(df['temp'] - 25) / 50
        temp_effect = np.clip(temp_effect, 0.2, 1.0)

        # Calculate count
        df['count'] = (
                base_count *
                df['season_mult'] *
                hour_effect *
                df['weather_mult'] *
                temp_effect *
                (2 if df['workingday'].iloc[0] else 1.2) *
                np.random.normal(1, 0.2, len(df))
        ).astype(int).clip(0, None)

        # Add some random noise
        df['count'] = df['count'] + np.random.poisson(5, len(df))
        df['count'] = df['count'].clip(0, None)

        return df

    def engineer_features(self, df):
        """Create additional features from the data"""

        df = df.copy()

        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['weekday'] = df['datetime'].dt.weekday

        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # Rush hour indicators (convert to int explicitly)
        df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                              ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)

        # Weekend indicator (convert to int explicitly)
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

        # Temperature interactions
        df['temp_humidity'] = df['temp'] * df['humidity'] / 100
        df['temp_windspeed'] = df['temp'] * df['windspeed']

        # Ensure all numeric columns are float
        numeric_columns = ['hour', 'day', 'month', 'year', 'weekday',
                           'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                           'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
                           'is_rush_hour', 'is_weekend', 'temp_humidity', 'temp_windspeed']

        for col in numeric_columns:
            df[col] = df[col].astype(np.float32)

        return df

    def prepare_data(self, df, target_col='count'):
        """Prepare data for training"""

        # Feature engineering
        df = self.engineer_features(df)

        # Select features
        feature_cols = [
            'season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'hour', 'day', 'month', 'weekday',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
            'is_rush_hour', 'is_weekend', 'temp_humidity', 'temp_windspeed'
        ]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Ensure all data is numeric and convert to float
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # Fill any NaN values that might have been created
        X = X.fillna(0)
        y = y.fillna(y.mean())

        # Convert boolean columns to int
        bool_cols = X.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)

        # Scale numerical features
        numerical_features = ['temp', 'atemp', 'humidity', 'windspeed', 'temp_humidity', 'temp_windspeed']

        self.scalers['numerical'] = StandardScaler()
        X[numerical_features] = self.scalers['numerical'].fit_transform(X[numerical_features])

        # Final conversion to numpy arrays with explicit dtype
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32)

        return X_array, y_array, feature_cols


class BikeForecasterTrainer:
    """Training manager for bike forecasting models"""

    def __init__(self, model_type='mlp', device='cpu'):
        self.device = torch.device(device)
        self.model_type = model_type
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}

    def build_model(self, input_size):
        """Build the forecasting model"""

        if self.model_type == 'mlp':
            self.model = BikeForecaster(input_size)
        elif self.model_type == 'lstm':
            self.model = LSTMForecaster(input_size)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.to(self.device)
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001):
        """Train the model"""

        # Create datasets
        train_dataset = BikeDataset(X_train, y_train)
        val_dataset = BikeDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Move data to the same device as model
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                if self.model_type == 'lstm':
                    # Add sequence dimension for LSTM
                    batch_X = batch_X.unsqueeze(1)

                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    # Move data to the same device as model
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if self.model_type == 'lstm':
                        batch_X = batch_X.unsqueeze(1)

                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_bike_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_bike_model.pth'))
        print(f'Training completed. Best validation loss: {best_val_loss:.4f}')

    def predict(self, X_test):
        """Make predictions"""

        self.model.eval()
        test_dataset = BikeDataset(X_test, np.zeros(len(X_test)))  # Dummy targets
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in test_loader:
                # Move data to the same device as model
                batch_X = batch_X.to(self.device)

                if self.model_type == 'lstm':
                    batch_X = batch_X.unsqueeze(1)

                outputs = self.model(batch_X).squeeze()
                # Move predictions back to CPU for numpy conversion
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def plot_training_history(self):
        """Plot training history"""

        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate model performance"""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_predictions(y_true, y_pred, title="Predictions vs Actual", sample_size=500):
    """Plot predictions vs actual values"""

    # Sample data for visualization if too large
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    ax1.scatter(y_true_sample, y_pred_sample, alpha=0.6)
    ax1.plot([y_true_sample.min(), y_true_sample.max()],
             [y_true_sample.min(), y_true_sample.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'{title} - Scatter Plot')
    ax1.grid(True)

    # Time series plot (first 100 points)
    n_points = min(100, len(y_true_sample))
    ax2.plot(range(n_points), y_true_sample[:n_points], label='Actual', marker='o')
    ax2.plot(range(n_points), y_pred_sample[:n_points], label='Predicted', marker='s')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Bike Count')
    ax2.set_title(f'{title} - Time Series')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""

    print("PyTorch Bike Demand Forecasting")
    print("=" * 50)

    # Initialize data processor
    processor = BikeDataProcessor()

    # Create sample data
    print("Creating sample data...")
    df = processor.create_sample_data(n_samples=8760)  # One year of hourly data

    # Prepare data
    print("Preparing features...")
    X, y, feature_names = processor.prepare_data(df)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}")

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train MLP model
    print("\nTraining MLP model...")
    mlp_trainer = BikeForecasterTrainer(model_type='mlp', device=device)
    mlp_model = mlp_trainer.build_model(input_size=X.shape[1])
    mlp_trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64)

    # Make predictions
    print("Making predictions...")
    y_pred_mlp = mlp_trainer.predict(X_test)

    # Evaluate model
    mlp_metrics = evaluate_model(y_test, y_pred_mlp, "MLP Model")

    # Plot results
    plot_predictions(y_test, y_pred_mlp, "MLP Model Predictions")
    mlp_trainer.plot_training_history()

    # Feature importance (simple version using gradients)
    print("\nCalculating feature importance...")
    mlp_model.eval()
    X_sample = torch.FloatTensor(X_test[:100]).to(device)
    X_sample.requires_grad_(True)

    output = mlp_model(X_sample).sum()
    output.backward()

    feature_importance = torch.abs(X_sample.grad).mean(dim=0).cpu().numpy()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), importance_df.head(10)['importance'])
    plt.yticks(range(10), importance_df.head(10)['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()