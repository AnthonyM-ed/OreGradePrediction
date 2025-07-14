# Temporary Data and Processing Cache

This directory contains temporary files and processing cache for improved performance.

## Directory Structure

- `processing_cache.pkl` - Cached data processing results
- `feature_cache.pkl` - Cached feature engineering outputs
- `spatial_cache.pkl` - Cached spatial analysis results
- `validation_cache.pkl` - Cached model validation results

## Cache Benefits

1. **Performance**: Avoid reprocessing identical data
2. **Consistency**: Ensure reproducible results
3. **Efficiency**: Reduce database load and computation time
4. **Scalability**: Handle large datasets efficiently

## Cache Management

Caches are automatically managed by the system:

- **TTL**: Time-to-live for cached items
- **Size limits**: Maximum cache size constraints
- **Cleanup**: Automatic cleanup of expired items
- **Invalidation**: Smart cache invalidation on data changes

## File Types

### processing_cache.pkl
- Preprocessed training data
- Feature engineering results
- Data validation outcomes

### feature_cache.pkl
- Spatial feature calculations
- Geological feature engineering
- Aggregated statistics

### spatial_cache.pkl
- Spatial interpolation results
- Neighbor analysis
- Grid generation outputs

## Usage

Cache files are automatically created and managed. Manual intervention is typically not required.

```python
from ml_models.cache.cache_manager import CacheManager

cache = CacheManager()
cached_data = cache.get_features(input_data, processing_config)
```

## Maintenance

- Cache files are automatically cleaned up
- Manual cleanup can be performed if needed
- Monitor cache size and performance impact

## Important Notes

- Cache files contain binary data (pickle format)
- Do not modify cache files manually
- Cache invalidation occurs automatically on data changes
- Regular cleanup maintains optimal performance
