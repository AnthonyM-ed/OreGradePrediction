# Model Training Results and Artifacts

This directory contains trained models, metadata, and evaluation results.

## Directory Structure

- `grade_model_v1.joblib` - Main trained XGBoost model
- `scaler.joblib` - Feature scaler for data normalization
- `model_metadata.json` - Model metadata and configuration
- `feature_importance.csv` - Feature importance rankings
- `training_history.csv` - Training history and metrics

## Model Information

The trained models are optimized for spatial ore grade prediction using XGBoost with hyperparameter tuning.

### Key Features:
- Spatial coordinates (latitude, longitude, depth)
- Geological features (element concentrations, ratios)
- Engineered features (spatial distances, grade continuity)
- Standardized grades in PPM

### Performance Metrics:
- RMSE: Root Mean Square Error
- R²: Coefficient of Determination
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error
- Geological-specific metrics (bias, cutoff accuracy)

## Usage

Models are automatically loaded by the prediction system:

```python
from ml_models.inference.predictor import SpatialOreGradePredictor

predictor = SpatialOreGradePredictor()
prediction = predictor.predict(
    latitude=-15.1915,
    longitude=-71.8358,
    depth=150.0,
    element='CU'
)
```

## Model Versions

- **v1**: Initial XGBoost model with basic spatial features
- **v2**: Enhanced model with advanced geological features (planned)
- **v3**: Ensemble model with multiple algorithms (planned)

## Backup and Versioning

Models are automatically backed up after training. Previous versions are maintained for comparison and rollback capabilities.
