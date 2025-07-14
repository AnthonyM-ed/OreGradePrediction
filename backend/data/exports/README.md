# Data Exports and Results

This directory contains exported data, predictions, and analysis results.

## Directory Structure

- `predictions.csv` - Generated predictions with confidence intervals
- `model_metrics.csv` - Model performance metrics across different runs
- `feature_importance.csv` - Feature importance rankings and scores
- `training_history.csv` - Training history and epoch-by-epoch metrics
- `spatial_predictions.csv` - Spatial grid predictions for mapping
- `validation_results.csv` - Cross-validation and test results

## Export Formats

### predictions.csv
Columns:
- `prediction_id`: Unique identifier
- `latitude`, `longitude`, `depth`: Spatial coordinates
- `element`: Element being predicted
- `predicted_grade_ppm`: Predicted grade in PPM
- `confidence_interval_lower`: Lower bound of confidence interval
- `confidence_interval_upper`: Upper bound of confidence interval
- `prediction_timestamp`: When prediction was made

### model_metrics.csv
Columns:
- `model_name`: Name of the model
- `training_date`: Date of training
- `rmse`: Root Mean Square Error
- `r2`: R-squared score
- `mae`: Mean Absolute Error
- `mape`: Mean Absolute Percentage Error
- `training_samples`: Number of training samples
- `validation_samples`: Number of validation samples

### feature_importance.csv
Columns:
- `feature_name`: Name of the feature
- `importance_score`: Importance score (0-1)
- `rank`: Ranking by importance
- `feature_type`: Type of feature (spatial, geological, engineered)

### spatial_predictions.csv
Columns:
- `grid_x`, `grid_y`: Grid coordinates
- `latitude`, `longitude`: Geographic coordinates
- `element`: Element predicted
- `predicted_grade_ppm`: Predicted grade
- `confidence_score`: Prediction confidence
- `prediction_date`: Date of prediction

## Usage

Export files are automatically generated during training and prediction:

```python
from ml_models.training.train_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.train_model(element='CU')
# Results automatically exported to CSV files
```

## Analysis and Visualization

Export files can be used for:
- Performance analysis and comparison
- Visualization in external tools
- Reporting and documentation
- Data validation and quality assurance

## File Management

- Files are timestamped for version control
- Automatic archiving of old results
- Size limits to prevent disk space issues
- Regular cleanup of outdated files

## Integration

Export files integrate with:
- Frontend visualization components
- External GIS systems
- Business intelligence tools
- Reporting dashboards

## Best Practices

1. **Regular Review**: Monitor export files for data quality
2. **Backup**: Keep backups of important prediction results
3. **Cleanup**: Remove outdated files to save space
4. **Documentation**: Maintain clear naming conventions
5. **Security**: Protect sensitive geological data appropriately
