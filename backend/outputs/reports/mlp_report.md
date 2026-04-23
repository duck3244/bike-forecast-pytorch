
# Model Performance Report
## Model: MLP Model

### Performance Metrics
- **R² Score**: 0.6654
- **RMSE**: 125.8535
- **MAE**: 86.9623
- **MSE**: 15839.1104
- **MAPE**: N/A

### Training Information
- **Training Time**: 9.25 seconds
- **Epochs**: 100
- **Batch Size**: 64
- **Learning Rate**: 0.001

### Model Configuration
```yaml
lstm:
  bidirectional: false
  dropout: 0.2
  hidden_size: 128
  num_layers: 2
mlp:
  activation: relu
  batch_norm: true
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

**Current Performance**: Poor

### Recommendations
- Consider increasing model complexity or adding more features
- Check for data quality issues or missing important features
- High RMSE indicates large prediction errors
- Consider ensemble methods or regularization techniques
    