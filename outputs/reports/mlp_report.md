
# Model Performance Report
## Model: MLP Model

### Performance Metrics
- **R² Score**: 0.8806
- **RMSE**: 61.8052
- **MAE**: 40.3787
- **MSE**: 3819.8794
- **MAPE**: N/A

### Training Information
- **Training Time**: 14.35 seconds
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
- High RMSE indicates large prediction errors
- Consider ensemble methods or regularization techniques
    
### Top 10 Most Important Features
     feature  importance
    hour_cos  110.217575
    hour_sin   89.449203
is_rush_hour   41.271667
        temp   29.622128
     holiday   27.997631
  workingday   24.861214
   month_sin   23.307827
       atemp   23.118357
     weather   20.369398
    humidity   20.324099
        