# ONNX Integration Guide

This guide explains how to integrate ONNX (Open Neural Network Exchange) models into your MQL5 Expert Advisors for native ML inference without Python dependencies.

## Overview

ONNX enables direct ML model execution in MQL5 with sub-5ms latency, eliminating the need for socket communication with Python. This provides significant performance improvements and reduces system complexity.

## Architecture

```
Python Training Environment          MQL5 Runtime Environment
┌─────────────────────────┐         ┌─────────────────────────┐
│ 1. Train ML Model       │         │ 4. Load ONNX Model     │
│    (TensorFlow/Keras)   │         │    (OnnxCreateFromFile)│
│                         │         │                         │
│ 2. Export to ONNX       │         │ 5. Prepare Features    │
│    (tf2onnx/skl2onnx)   │         │    (CalculateFeatures) │
│                         │         │                         │
│ 3. Save Model File      │◄────────┤ 6. Run Inference       │
│    (*.onnx)             │  Copy   │    (OnnxRun)           │
└─────────────────────────┘         └─────────────────────────┘
```

## Supported Models

### 1. Random Forest (scikit-learn)
```python
from advanced_models.models import RandomForestPredictor

model = RandomForestPredictor(n_estimators=100, max_depth=10)
model.train(X_train, y_train)
model.export_to_onnx(sample_input, "rf_model.onnx")
```

### 2. XGBoost
```python
from advanced_models.models import XGBoostPredictor

model = XGBoostPredictor(n_estimators=100, max_depth=6, learning_rate=0.1)
model.train(X_train, y_train)
model.export_to_onnx(sample_input, "xgb_model.onnx")
```

### 3. LSTM (TensorFlow/Keras)
```python
from advanced_models.models import LSTMPredictor

model = LSTMPredictor(units=64, dropout=0.2, epochs=50)
model.train(X_train, y_train)
model.export_to_onnx(sample_input, "lstm_model.onnx")
```

## MQL5 Integration

### Loading ONNX Models

```mq5
// Global variables
long onnxHandle = INVALID_HANDLE;
matrix inputMatrix;
matrix outputMatrix;

// Initialize ONNX model
bool InitializeONNX()
{
    // Load ONNX model
    onnxHandle = OnnxCreateFromFile("rf_model.onnx");
    if (onnxHandle == INVALID_HANDLE)
    {
        Print("Failed to load ONNX model");
        return false;
    }

    // Initialize matrices (1 sample, 50 features)
    inputMatrix.Resize(1, 50);
    outputMatrix.Resize(1, 1);  // Binary classification

    return true;
}
```

### Feature Calculation

```mq5
bool CalculateFeatures(matrix &features)
{
    // Price data
    double close = iClose(Symbol(), PERIOD_M15, 0);
    double high = iHigh(Symbol(), PERIOD_M15, 0);
    double low = iLow(Symbol(), PERIOD_M15, 0);
    double open = iOpen(Symbol(), PERIOD_M15, 0);

    // Basic features
    features[0][0] = close;
    features[0][1] = (close - open) / open;  // Body size
    features[0][2] = (high - low) / close;   // Range

    // Technical indicators
    features[0][3] = iMA(Symbol(), PERIOD_M15, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
    features[0][4] = iMA(Symbol(), PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE, 0);
    features[0][5] = iRSI(Symbol(), PERIOD_M15, 14, PRICE_CLOSE, 0);
    features[0][6] = iATR(Symbol(), PERIOD_M15, 14, 0);

    // Add more features up to 50...

    return true;
}
```

### Running Inference

```mq5
int GenerateMLSignal()
{
    if (onnxHandle == INVALID_HANDLE)
        return 0;

    // Calculate features
    if (!CalculateFeatures(inputMatrix))
        return 0;

    // Run ONNX inference
    if (!OnnxRun(onnxHandle, ONNX_RUN_INPUT, inputMatrix, ONNX_RUN_OUTPUT, outputMatrix))
    {
        Print("ONNX inference failed");
        return 0;
    }

    // Get prediction
    double prediction = outputMatrix[0][0];

    // Apply threshold
    if (prediction >= 0.6)      // Bullish threshold
        return 1;
    else if (prediction <= 0.4) // Bearish threshold
        return -1;

    return 0; // Neutral
}
```

## Training Pipeline

### Automated ONNX Export

```python
from onnx_training_pipeline import ONNXTrainingPipeline

# Train and export Random Forest model
pipeline = ONNXTrainingPipeline("EURUSD", "random_forest")
pipeline.run_pipeline()

# Files created:
# - models/EURUSD_random_forest.onnx (ONNX model)
# - models/EURUSD_random_forest_scaler.pkl (Feature scaler)
```

### Command Line Training

```bash
# Train Random Forest for EURUSD
python onnx_training_pipeline.py --symbol EURUSD --model random_forest

# Train XGBoost for GBPUSD
python onnx_training_pipeline.py --symbol GBPUSD --model xgboost

# Train LSTM for USDJPY
python onnx_training_pipeline.py --symbol USDJPY --model lstm
```

## Performance Optimization

### Model Quantization

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load and quantize model
model_fp32 = 'model.onnx'
model_quant = 'model_quant.onnx'

quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
```

### Feature Caching

```mq5
// Cache features to avoid recalculation
matrix cachedFeatures;
datetime lastFeatureTime = 0;

bool GetFeatures(matrix &features)
{
    datetime currentTime = iTime(Symbol(), PERIOD_M15, 0);

    if (currentTime != lastFeatureTime)
    {
        CalculateFeatures(cachedFeatures);
        lastFeatureTime = currentTime;
    }

    features = cachedFeatures;
    return true;
}
```

## Error Handling

### Model Loading Errors

```mq5
bool LoadONNXModel(string modelPath)
{
    onnxHandle = OnnxCreateFromFile(modelPath);
    if (onnxHandle == INVALID_HANDLE)
    {
        Print("Failed to load ONNX model: ", modelPath);
        Print("Error: ", GetLastError());
        return false;
    }

    // Validate model
    OnnxTypeInfo typeInfo;
    if (!OnnxGetModelTypeInfo(onnxHandle, typeInfo))
    {
        Print("Invalid ONNX model format");
        OnnxRelease(onnxHandle);
        return false;
    }

    return true;
}
```

### Inference Errors

```mq5
int SafeInference()
{
    if (onnxHandle == INVALID_HANDLE)
        return 0;

    matrix features;
    if (!GetFeatures(features))
        return 0;

    matrix output;
    output.Resize(1, 1);

    if (!OnnxRun(onnxHandle, ONNX_RUN_INPUT, features, ONNX_RUN_OUTPUT, output))
    {
        Print("ONNX inference failed, falling back to simple strategy");
        return SimpleStrategy();  // Fallback strategy
    }

    return (int)output[0][0];
}
```

## SMC Strategy Integration

### Fractal Pattern Detection

```mq5
bool CheckFractalPattern(int signal)
{
    int direction = (signal > 0) ? 1 : -1;

    for (int i = 1; i <= 5; i++)  // 5-bar fractal
    {
        double high1 = iHigh(Symbol(), PERIOD_M15, i);
        double high2 = iHigh(Symbol(), PERIOD_M15, i+1);
        double low1 = iLow(Symbol(), PERIOD_M15, i);
        double low2 = iLow(Symbol(), PERIOD_M15, i+1);

        // Bearish fractal (resistance sweep)
        if (direction == -1 && high1 > high2 && high1 > iHigh(Symbol(), PERIOD_M15, i-1))
            return true;

        // Bullish fractal (support sweep)
        if (direction == 1 && low1 < low2 && low1 < iLow(Symbol(), PERIOD_M15, i-1))
            return true;
    }

    return false;
}
```

### Order Block Detection

```mq5
bool CheckOrderBlocks(int signal)
{
    // Large candle followed by small candle
    double body1 = MathAbs(iOpen(Symbol(), PERIOD_M15, 1) - iClose(Symbol(), PERIOD_M15, 1));
    double body2 = MathAbs(iOpen(Symbol(), PERIOD_M15, 0) - iClose(Symbol(), PERIOD_M15, 0));

    double range1 = iHigh(Symbol(), PERIOD_M15, 1) - iLow(Symbol(), PERIOD_M15, 1);
    double range2 = iHigh(Symbol(), PERIOD_M15, 0) - iLow(Symbol(), PERIOD_M15, 0);

    // Large candle followed by small candle
    if (body1 > range1 * 0.6 && body2 < range2 * 0.3)
    {
        bool bullishOB = (iClose(Symbol(), PERIOD_M15, 1) > iOpen(Symbol(), PERIOD_M15, 1) && signal > 0);
        bool bearishOB = (iClose(Symbol(), PERIOD_M15, 1) < iOpen(Symbol(), PERIOD_M15, 1) && signal < 0);

        if (bullishOB || bearishOB)
            return true;
    }

    return false;
}
```

## Deployment Checklist

### Pre-Deployment
- [ ] Train models with sufficient historical data
- [ ] Export models to ONNX format
- [ ] Test ONNX models with onnxruntime
- [ ] Validate feature calculations match Python
- [ ] Test inference latency (< 5ms target)

### MQL5 Setup
- [ ] Copy ONNX model to MQL5 Files folder
- [ ] Enable automated trading in MT5
- [ ] Configure EA parameters
- [ ] Test with demo account first
- [ ] Monitor initial trades closely

### Production Monitoring
- [ ] Set up performance logging
- [ ] Configure alert thresholds
- [ ] Monitor system resources
- [ ] Regular model retraining schedule
- [ ] Backup models and configurations

## Troubleshooting

### Common Issues

#### Model Loading Fails
```
Error: Failed to load ONNX model
```
**Solutions:**
- Verify model file path is correct
- Check MT5 file permissions
- Ensure ONNX model is not corrupted
- Try different model format (float32 vs quantized)

#### Inference Performance Issues
```
Warning: ONNX inference > 10ms
```
**Solutions:**
- Use model quantization
- Reduce feature count
- Cache features between bars
- Optimize matrix operations

#### Feature Mismatch
```
Error: Input tensor shape mismatch
```
**Solutions:**
- Verify feature count matches model (50 features)
- Check data types (float32)
- Ensure proper matrix initialization
- Debug feature calculation step-by-step

### Performance Benchmarks

| Model Type | Training Time | Model Size | Inference Time | Accuracy |
|------------|---------------|------------|----------------|----------|
| Random Forest | 2-5 min | 5-20 MB | 1-2 ms | 65-75% |
| XGBoost | 5-15 min | 10-50 MB | 2-3 ms | 70-80% |
| LSTM | 20-60 min | 20-100 MB | 3-5 ms | 75-85% |

### Memory Usage

- ONNX Runtime: ~50-200 MB per model
- Feature matrices: ~1-2 KB per inference
- Model cache: Varies by model size
- Total RAM: 500MB+ recommended

## Advanced Topics

### Multi-Model Ensemble

```mq5
int EnsemblePrediction()
{
    double rf_pred = RunONNXInference("rf_model.onnx");
    double xgb_pred = RunONNXInference("xgb_model.onnx");
    double lstm_pred = RunONNXInference("lstm_model.onnx");

    // Weighted voting
    double ensemble = (rf_pred * 0.4) + (xgb_pred * 0.4) + (lstm_pred * 0.2);

    return ensemble > 0.5 ? 1 : -1;
}
```

### Dynamic Model Switching

```mq5
string SelectBestModel()
{
    // Market regime detection
    double volatility = CalculateVolatility();
    double trend = CalculateTrendStrength();

    if (volatility > 0.8)
        return "lstm_model.onnx";      // Neural networks for volatile markets
    else if (trend > 0.7)
        return "xgb_model.onnx";       // Boosting for trending markets
    else
        return "rf_model.onnx";        // Forests for ranging markets
}
```

### Online Learning Integration

```mq5
void UpdateModelWithNewData()
{
    // Collect recent trade outcomes
    // Send data back to Python for model updates
    // Download updated ONNX model
    // Reload model in MQL5
}
```

## Resources

### Documentation
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c++)
- [MQL5 Matrix Operations](https://www.mql5.com/en/docs/matrix)
- [MetaTrader 5 ONNX Support](https://www.mql5.com/en/docs/onnx)

### Tools
- [Netron](https://netron.app/) - ONNX model visualization
- [ONNX Runtime](https://onnxruntime.ai/) - Inference engine
- [tf2onnx](https://github.com/onnx/tensorflow-onnx) - TensorFlow export

### Community
- [MQL5 ONNX Forum](https://www.mql5.com/en/forum)
- [ONNX GitHub](https://github.com/onnx/onnx)
- [QuantConnect](https://www.quantconnect.com/) - Alternative platforms

---

**Note:** ONNX integration requires MetaTrader 5 build 3600+ and proper model training. Always test thoroughly in demo environment before live trading.