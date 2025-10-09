"""
Data Exploration Script for Bike Demand Forecasting

This script provides comprehensive exploratory data analysis for bike sharing demand forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pathlib import Path
import argparse
from scipy import stats

warnings.filterwarnings('ignore')

# Import our custom classes
from bike_forecast_pytorch import BikeDataProcessor

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BikeDataExplorer:
    """Class for exploring bike sharing data"""
    
    def __init__(self, output_dir="outputs/exploration"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = BikeDataProcessor()
        
    def load_or_generate_data(self, data_path=None, n_samples=8760):
        """Load data or generate synthetic data"""
        if data_path and Path(data_path).exists():
            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            if 'datetime' not in df.columns:
                df['datetime'] = pd.date_range(start='2011-01-01', periods=len(df), freq='H')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            print("Generating synthetic bike sharing data...")
            df = self.processor.create_sample_data(n_samples=n_samples)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    
    def basic_statistics(self, df):
        """Generate and display basic statistics"""
        print("\n" + "="*60)
        print("BASIC DATASET INFORMATION")
        print("="*60)
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nDescriptive Statistics:")
        print(df.describe())
        
        print("\nMissing Values:")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found!")
        
        return df.describe()
    
    def analyze_target_variable(self, df, save_plots=True):
        """Analyze the target variable (bike count)"""
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram
        axes[0, 0].hist(df['count'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Bike Counts')
        axes[0, 0].set_xlabel('Count')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(df['count'])
        axes[0, 1].set_title('Box Plot of Bike Counts')
        axes[0, 1].set_ylabel('Count')
        
        # Time series plot (first 168 hours = 1 week)
        week_data = df.head(168)
        axes[1, 0].plot(week_data['datetime'], week_data['count'], marker='o', markersize=3)
        axes[1, 0].set_title('Bike Counts - First Week')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # QQ plot
        stats.probplot(df['count'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'target_variable_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"Mean bike count: {df['count'].mean():.2f}")
        print(f"Median bike count: {df['count'].median():.2f}")
        print(f"Standard deviation: {df['count'].std():.2f}")
        print(f"Min count: {df['count'].min()}")
        print(f"Max count: {df['count'].max()}")
        print(f"Skewness: {df['count'].skew():.2f}")
        print(f"Kurtosis: {df['count'].kurtosis():.2f}")
    
    def analyze_temporal_patterns(self, df, save_plots=True):
        """Analyze temporal patterns in the data"""
        print("\n" + "="*60)
        print("TEMPORAL PATTERNS ANALYSIS")
        print("="*60)
        
        # Add temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average by hour
        hourly_avg = df.groupby('hour')['count'].agg(['mean', 'std']).reset_index()
        axes[0, 0].plot(hourly_avg['hour'], hourly_avg['mean'], marker='o', linewidth=2)
        axes[0, 0].fill_between(hourly_avg['hour'], 
                                hourly_avg['mean'] - hourly_avg['std'],
                                hourly_avg['mean'] + hourly_avg['std'], 
                                alpha=0.3)
        axes[0, 0].set_title('Average Bike Count by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Count')
        axes[0, 0].grid(True)
        
        # Day of week patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = df.groupby('day_of_week')['count'].mean().reindex(day_order)
        axes[0, 1].bar(range(len(daily_avg)), daily_avg.values, color='skyblue')
        axes[0, 1].set_title('Average Bike Count by Day of Week')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Count')
        axes[0, 1].set_xticks(range(len(day_order)))
        axes[0, 1].set_xticklabels(day_order, rotation=45)
        
        # Monthly patterns
        monthly_avg = df.groupby('month')['count'].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
        axes[1, 0].set_title('Average Bike Count by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Count')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].grid(True)
        
        # Seasonal patterns
        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        seasonal_avg = df.groupby('season')['count'].mean()
        season_labels = [season_names[i] for i in seasonal_avg.index]
        axes[1, 1].bar(season_labels, seasonal_avg.values, color=['green', 'red', 'orange', 'blue'])
        axes[1, 1].set_title('Average Bike Count by Season')
        axes[1, 1].set_xlabel('Season')
        axes[1, 1].set_ylabel('Average Count')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print insights
        peak_hours = hourly_avg.nlargest(3, 'mean')['hour'].tolist()
        print(f"Peak usage hours: {', '.join(map(str, peak_hours))}")
        
        return hourly_avg, daily_avg, monthly_avg, seasonal_avg
    
    def analyze_weather_impact(self, df, save_plots=True):
        """Analyze weather impact on bike usage"""
        print("\n" + "="*60)
        print("WEATHER IMPACT ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Weather conditions
        weather_names = {1: 'Clear', 2: 'Mist', 3: 'Light Rain', 4: 'Heavy Rain'}
        weather_avg = df.groupby('weather')['count'].mean()
        weather_labels = [weather_names[i] for i in weather_avg.index]
        axes[0, 0].bar(weather_labels, weather_avg.values, 
                       color=['gold', 'lightblue', 'lightgray', 'darkblue'])
        axes[0, 0].set_title('Average Bike Count by Weather Condition')
        axes[0, 0].set_ylabel('Average Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Temperature vs count
        axes[0, 1].scatter(df['temp'], df['count'], alpha=0.6, s=1)
        axes[0, 1].set_title('Temperature vs Bike Count')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Count')
        
        # Humidity vs count
        axes[0, 2].scatter(df['humidity'], df['count'], alpha=0.6, s=1)
        axes[0, 2].set_title('Humidity vs Bike Count')
        axes[0, 2].set_xlabel('Humidity (%)')
        axes[0, 2].set_ylabel('Count')
        
        # Wind speed vs count
        axes[1, 0].scatter(df['windspeed'], df['count'], alpha=0.6, s=1)
        axes[1, 0].set_title('Wind Speed vs Bike Count')
        axes[1, 0].set_xlabel('Wind Speed')
        axes[1, 0].set_ylabel('Count')
        
        # Working day vs holiday
        workday_data = df.groupby(['workingday', 'holiday'])['count'].mean().unstack(fill_value=0)
        workday_data.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average Count: Working Day vs Holiday')
        axes[1, 1].set_xlabel('Working Day (0=No, 1=Yes)')
        axes[1, 1].set_ylabel('Average Count')
        axes[1, 1].legend(['Not Holiday', 'Holiday'])
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        # Temperature distribution by season
        for season in df['season'].unique():
            season_data = df[df['season'] == season]['temp']
            axes[1, 2].hist(season_data, alpha=0.7, label=weather_names.get(season, f'Season {season}'), bins=20)
        axes[1, 2].set_title('Temperature Distribution by Season')
        axes[1, 2].set_xlabel('Temperature (°C)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'weather_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print insights
        best_weather = weather_avg.idxmax()
        print(f"Best weather condition: {weather_names[best_weather]}")
        
        temp_corr = df['temp'].corr(df['count'])
        print(f"Temperature correlation with usage: {temp_corr:.3f}")
        
        return weather_avg
    
    def analyze_correlations(self, df, save_plots=True):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Add hour and month for correlation analysis
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        
        # Calculate correlation matrix
        numeric_columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 
                           'atemp', 'humidity', 'windspeed', 'count', 'hour', 'month']
        corr_matrix = df[numeric_columns].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Features')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print strongest correlations with count
        count_corr = corr_matrix['count'].abs().sort_values(ascending=False)
        print("Strongest correlations with bike count:")
        print(count_corr.head(10))
        
        return corr_matrix
    
    def analyze_feature_engineering(self, df, save_plots=True):
        """Preview feature engineering results"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PREVIEW")
        print("="*60)
        
        # Apply feature engineering
        df_engineered = self.processor.engineer_features(df)
        
        print(f"Original features: {len(df.columns)}")
        print(f"Engineered features: {len(df_engineered.columns)}")
        print("\nNew features added:")
        new_features = set(df_engineered.columns) - set(df.columns)
        for feature in sorted(new_features):
            print(f"  - {feature}")
        
        # Visualize cyclical features
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hour cyclical encoding
        scatter = axes[0, 0].scatter(df_engineered['hour_sin'], df_engineered['hour_cos'], 
                                   c=df_engineered['hour'], cmap='viridis', s=20)
        axes[0, 0].set_title('Hour Cyclical Encoding')
        axes[0, 0].set_xlabel('Hour Sin')
        axes[0, 0].set_ylabel('Hour Cos')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Month cyclical encoding  
        scatter = axes[0, 1].scatter(df_engineered['month_sin'], df_engineered['month_cos'], 
                                   c=df_engineered['month'], cmap='viridis', s=20)
        axes[0, 1].set_title('Month Cyclical Encoding')
        axes[0, 1].set_xlabel('Month Sin')
        axes[0, 1].set_ylabel('Month Cos')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Rush hour analysis
        rush_hour_avg = df_engineered.groupby('is_rush_hour')['count'].mean()
        axes[1, 0].bar(['Non-Rush', 'Rush Hour'], rush_hour_avg.values, 
                       color=['lightcoral', 'steelblue'])
        axes[1, 0].set_title('Average Count: Rush vs Non-Rush Hours')
        axes[1, 0].set_ylabel('Average Count')
        
        # Weekend analysis
        weekend_avg = df_engineered.groupby('is_weekend')['count'].mean()
        axes[1, 1].bar(['Weekday', 'Weekend'], weekend_avg.values, 
                       color=['lightgreen', 'orange'])
        axes[1, 1].set_title('Average Count: Weekday vs Weekend')
        axes[1, 1].set_ylabel('Average Count')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'feature_engineering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_engineered
    
    def detect_outliers(self, df, save_plots=True):
        """Detect and visualize outliers"""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        def detect_outliers_iqr(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        # Check for outliers in key features
        outlier_features = ['count', 'temp', 'humidity', 'windspeed']
        outlier_summary = {}
        
        for feature in outlier_features:
            outliers = detect_outliers_iqr(df[feature])
            outlier_summary[feature] = {
                'count': outliers.sum(),
                'percentage': (outliers.sum() / len(df)) * 100
            }
        
        print("Outlier Detection Summary:")
        print("-" * 40)
        for feature, stats in outlier_summary.items():
            print(f"{feature}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        
        # Visualize outliers
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(outlier_features):
            axes[i].boxplot(df[feature])
            axes[i].set_title(f'{feature.title()} - Outlier Detection')
            axes[i].set_ylabel(feature.title())
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return outlier_summary
    
    def generate_insights(self, df, hourly_avg, seasonal_avg, weather_avg, weekend_avg):
        """Generate key insights from the analysis"""
        print("\n" + "="*60)
        print("KEY INSIGHTS SUMMARY")
        print("="*60)
        
        insights = []
        
        # Peak hours
        peak_hours = hourly_avg.nlargest(3, 'mean')['hour'].tolist()
        insights.append(f"Peak usage hours: {', '.join(map(str, peak_hours))}")
        
        # Best weather for biking
        weather_names = {1: 'Clear', 2: 'Mist', 3: 'Light Rain', 4: 'Heavy Rain'}
        best_weather = weather_avg.idxmax()
        insights.append(f"Best weather condition: {weather_names[best_weather]}")
        
        # Seasonal preferences
        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        best_season = seasonal_avg.idxmax()
        insights.append(f"Most popular season: {season_names[best_season]}")
        
        # Temperature correlation
        temp_corr = df['temp'].corr(df['count'])
        insights.append(f"Temperature correlation with usage: {temp_corr:.3f}")
        
        # Weekend vs weekday
        weekend_diff = weekend_avg[True] - weekend_avg[False]
        weekend_higher = "higher" if weekend_diff > 0 else "lower"
        insights.append(f"Weekend usage is {abs(weekend_diff):.1f} bikes {weekend_higher} than weekdays")
        
        print("Key Insights from Data Exploration:")
        print("=" * 50)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\nData is ready for model training!")
        
        # Save insights to file
        with open(self.output_dir / 'insights_summary.txt', 'w') as f:
            f.write("Key Insights from Bike Demand Data Exploration\n")
            f.write("=" * 50 + "\n\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
        
        return insights
    
    def run_complete_analysis(self, data_path=None, n_samples=8760, save_plots=True):
        """Run the complete data exploration pipeline"""
        print("Starting Comprehensive Bike Demand Data Exploration")
        print("=" * 60)
        
        # Load data
        df = self.load_or_generate_data(data_path, n_samples)
        
        # Basic statistics
        self.basic_statistics(df)
        
        # Target variable analysis
        self.analyze_target_variable(df, save_plots)
        
        # Temporal patterns
        hourly_avg, daily_avg, monthly_avg, seasonal_avg = self.analyze_temporal_patterns(df, save_plots)
        
        # Weather impact
        weather_avg = self.analyze_weather_impact(df, save_plots)
        
        # Correlation analysis
        self.analyze_correlations(df, save_plots)
        
        # Feature engineering
        self.analyze_feature_engineering(df, save_plots)
        
        # Data quality
        self.detect_outliers(df, save_plots)
        
        # Generate insights
        weekend_avg = df.groupby(df['datetime'].dt.weekday.isin([5, 6]))['count'].mean()
        insights = self.generate_insights(df, hourly_avg, seasonal_avg, weather_avg, weekend_avg)
        
        print(f"\n{'='*60}")
        print("DATA EXPLORATION COMPLETED!")
        print(f"All plots and reports saved to: {self.output_dir}")
        print(f"{'='*60}")
        
        return df, insights

def main():
    """Main function for running data exploration"""
    parser = argparse.ArgumentParser(description='Bike Demand Data Exploration')
    parser.add_argument('--data', type=str, help='Path to bike sharing dataset')
    parser.add_argument('--output-dir', type=str, default='outputs/exploration', 
                        help='Output directory for plots and reports')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--samples', type=int, default=8760, 
                        help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = BikeDataExplorer(output_dir=args.output_dir)
    
    # Run analysis
    df, insights = explorer.run_complete_analysis(
        data_path=args.data,
        n_samples=args.samples,
        save_plots=not args.no_plots
    )

if __name__ == "__main__":
    main()