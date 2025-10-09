
# Model Performance Report
## Model: mlp

### Performance Metrics
- **R² Score**: 0.8695
- **RMSE**: 20.0188
- **MAE**: 13.6944
- **MSE**: 400.7513
- **MAPE**: 18.49%

### Training Information
- **Training Time**: 5.32 seconds
- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 0.001

### Model Configuration
```yaml
mlp:
  dropout_rate: 0.2
  hidden_sizes:
  - 512
  - 256
  - 128
  - 64

```

### Performance Interpretation
- **Excellent**: R² > 0.9, RMSE < 10
- **Good**: R² > 0.8, RMSE < 20
- **Fair**: R² > 0.7, RMSE < 30
- **Poor**: R² < 0.7, RMSE > 30

**Current Performance**: Fair

### Recommendations
- Performance is acceptable for most use cases
- Continue monitoring model performance on new data
    