import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from scipy import stats
import json
import os
from django.conf import settings

logger = logging.getLogger(__name__)

class GeologicalDataPreprocessor:
    """Preprocesador de datos geológicos"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.outlier_bounds = {}
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Cargar configuración de preprocesamiento"""
        try:
            config_path = os.path.join(settings.BASE_DIR, 'config', 'ml_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('preprocessing', {})
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuración por defecto"""
        return {
            "handle_missing": "median",
            "outlier_detection": "iqr",
            "scaling": "standard",
            "feature_selection": "correlation"
        }
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = None) -> pd.DataFrame:
        """
        Manejar valores faltantes
        
        Args:
            df: DataFrame con datos
            strategy: Estrategia para manejar valores faltantes
            
        Returns:
            DataFrame con valores faltantes manejados
        """
        if strategy is None:
            strategy = self.config.get('handle_missing', 'median')
        
        df_processed = df.copy()
        
        # Separar columnas numéricas y categóricas
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Manejar valores faltantes en columnas numéricas
        if len(numeric_columns) > 0:
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy='median')
            
            df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])
            self.imputers['numeric'] = imputer
        
        # Manejar valores faltantes en columnas categóricas
        if len(categorical_columns) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_columns] = imputer_cat.fit_transform(df_processed[categorical_columns])
            self.imputers['categorical'] = imputer_cat
        
        logger.info(f"Valores faltantes manejados usando estrategia: {strategy}")
        return df_processed
    
    def detect_outliers(self, df: pd.DataFrame, method: str = None) -> Dict[str, List]:
        """
        Detectar valores atípicos
        
        Args:
            df: DataFrame con datos
            method: Método de detección ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            Diccionario con índices de outliers por columna
        """
        if method is None:
            method = self.config.get('outlier_detection', 'iqr')
        
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if method == 'iqr':
                outliers[col] = self._detect_outliers_iqr(df[col])
            elif method == 'zscore':
                outliers[col] = self._detect_outliers_zscore(df[col])
            else:
                outliers[col] = self._detect_outliers_iqr(df[col])
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        logger.info(f"Detectados {total_outliers} outliers usando método: {method}")
        
        return outliers
    
    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detectar outliers usando IQR"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.outlier_bounds[series.name] = {
            'lower': lower_bound,
            'upper': upper_bound,
            'method': 'iqr'
        }
        
        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outliers
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detectar outliers usando Z-score"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = series[z_scores > threshold].index.tolist()
        
        self.outlier_bounds[series.name] = {
            'threshold': threshold,
            'method': 'zscore'
        }
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap', 
                       detection_method: str = None) -> pd.DataFrame:
        """
        Manejar valores atípicos
        
        Args:
            df: DataFrame con datos
            method: Método para manejar outliers ('cap', 'remove', 'transform')
            detection_method: Método de detección
            
        Returns:
            DataFrame con outliers manejados
        """
        df_processed = df.copy()
        outliers = self.detect_outliers(df_processed, detection_method)
        
        if method == 'cap':
            df_processed = self._cap_outliers(df_processed, outliers)
        elif method == 'remove':
            df_processed = self._remove_outliers(df_processed, outliers)
        elif method == 'transform':
            df_processed = self._transform_outliers(df_processed)
        
        logger.info(f"Outliers manejados usando método: {method}")
        return df_processed
    
    def _cap_outliers(self, df: pd.DataFrame, outliers: Dict[str, List]) -> pd.DataFrame:
        """Limitar outliers a los bounds"""
        df_capped = df.copy()
        
        for col, bounds in self.outlier_bounds.items():
            if col in df_capped.columns and bounds['method'] == 'iqr':
                df_capped[col] = df_capped[col].clip(
                    lower=bounds['lower'], 
                    upper=bounds['upper']
                )
        
        return df_capped
    
    def _remove_outliers(self, df: pd.DataFrame, outliers: Dict[str, List]) -> pd.DataFrame:
        """Remover filas con outliers"""
        # Combinar todos los índices de outliers
        all_outlier_indices = set()
        for indices in outliers.values():
            all_outlier_indices.update(indices)
        
        # Remover filas con outliers
        df_clean = df.drop(index=list(all_outlier_indices))
        
        logger.info(f"Removidas {len(all_outlier_indices)} filas con outliers")
        return df_clean
    
    def _transform_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transformar datos para reducir impacto de outliers"""
        df_transformed = df.copy()
        numeric_columns = df_transformed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Aplicar transformación log para datos positivos
            if (df_transformed[col] > 0).all():
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
            
            # Aplicar transformación sqrt
            if (df_transformed[col] >= 0).all():
                df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col])
        
        return df_transformed
    
    def scale_features(self, df: pd.DataFrame, method: str = None, 
                      fit: bool = True) -> pd.DataFrame:
        """
        Escalar características
        
        Args:
            df: DataFrame con datos
            method: Método de escalado ('standard', 'robust', 'minmax')
            fit: Si ajustar el escalador
            
        Returns:
            DataFrame con características escaladas
        """
        if method is None:
            method = self.config.get('scaling', 'standard')
        
        df_scaled = df.copy()
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return df_scaled
        
        # Seleccionar escalador
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        if fit:
            df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
            self.scalers[method] = scaler
        else:
            if method in self.scalers:
                df_scaled[numeric_columns] = self.scalers[method].transform(df_scaled[numeric_columns])
            else:
                logger.warning(f"Escalador {method} no encontrado. Ajustando nuevo escalador.")
                df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
                self.scalers[method] = scaler
        
        logger.info(f"Características escaladas usando método: {method}")
        return df_scaled
    
    def remove_low_variance_features(self, df: pd.DataFrame, 
                                   threshold: float = 0.01) -> pd.DataFrame:
        """
        Remover características con baja varianza
        
        Args:
            df: DataFrame con datos
            threshold: Umbral de varianza
            
        Returns:
            DataFrame sin características de baja varianza
        """
        df_filtered = df.copy()
        numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
        
        low_variance_columns = []
        for col in numeric_columns:
            if df_filtered[col].var() < threshold:
                low_variance_columns.append(col)
        
        if low_variance_columns:
            df_filtered = df_filtered.drop(columns=low_variance_columns)
            logger.info(f"Removidas {len(low_variance_columns)} características con baja varianza")
        
        return df_filtered
    
    def remove_highly_correlated_features(self, df: pd.DataFrame, 
                                        threshold: float = 0.95) -> pd.DataFrame:
        """
        Remover características altamente correlacionadas
        
        Args:
            df: DataFrame con datos
            threshold: Umbral de correlación
            
        Returns:
            DataFrame sin características altamente correlacionadas
        """
        df_filtered = df.copy()
        numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return df_filtered
        
        # Calcular matriz de correlación
        correlation_matrix = df_filtered[numeric_columns].corr().abs()
        
        # Encontrar pares altamente correlacionados
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] >= threshold:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))
        
        # Remover una característica de cada par altamente correlacionado
        columns_to_remove = set()
        for col1, col2, corr in high_corr_pairs:
            # Remover la característica con menor varianza
            if df_filtered[col1].var() < df_filtered[col2].var():
                columns_to_remove.add(col1)
            else:
                columns_to_remove.add(col2)
        
        if columns_to_remove:
            df_filtered = df_filtered.drop(columns=list(columns_to_remove))
            logger.info(f"Removidas {len(columns_to_remove)} características altamente correlacionadas")
        
        return df_filtered
    
    def detect_data_drift(self, df_reference: pd.DataFrame, 
                         df_new: pd.DataFrame, 
                         method: str = 'ks_test') -> Dict[str, float]:
        """
        Detectar deriva de datos (data drift)
        
        Args:
            df_reference: DataFrame de referencia
            df_new: DataFrame nuevo para comparar
            method: Método de detección ('ks_test', 'chi2_test')
            
        Returns:
            Diccionario con p-valores por característica
        """
        drift_results = {}
        numeric_columns = df_reference.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df_new.columns:
                if method == 'ks_test':
                    # Test de Kolmogorov-Smirnov
                    statistic, p_value = stats.ks_2samp(
                        df_reference[col].dropna(), 
                        df_new[col].dropna()
                    )
                    drift_results[col] = p_value
                else:
                    # Usar KS test como default
                    statistic, p_value = stats.ks_2samp(
                        df_reference[col].dropna(), 
                        df_new[col].dropna()
                    )
                    drift_results[col] = p_value
        
        logger.info(f"Análisis de deriva completado para {len(drift_results)} características")
        return drift_results
    
    def create_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Crear reporte de calidad de datos
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con métricas de calidad
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Análisis de valores faltantes
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        
        # Estadísticas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            report['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        # Estadísticas categóricas
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            unique_values = df[col].nunique()
            most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
            
            report['categorical_stats'][col] = {
                'unique_values': int(unique_values),
                'most_common': most_common,
                'most_common_count': int(df[col].value_counts().iloc[0]) if unique_values > 0 else 0
            }
        
        logger.info("Reporte de calidad de datos creado")
        return report
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                           handle_missing: bool = True,
                           handle_outliers: bool = True,
                           scale_features: bool = True,
                           remove_low_variance: bool = True,
                           remove_correlated: bool = True) -> pd.DataFrame:
        """
        Pipeline completo de preprocesamiento
        
        Args:
            df: DataFrame a procesar
            handle_missing: Si manejar valores faltantes
            handle_outliers: Si manejar outliers
            scale_features: Si escalar características
            remove_low_variance: Si remover características de baja varianza
            remove_correlated: Si remover características correlacionadas
            
        Returns:
            DataFrame procesado
        """
        logger.info("Iniciando pipeline de preprocesamiento")
        
        df_processed = df.copy()
        
        # Manejar valores faltantes
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
        
        # Manejar outliers
        if handle_outliers:
            df_processed = self.handle_outliers(df_processed, method='cap')
        
        # Remover características de baja varianza
        if remove_low_variance:
            df_processed = self.remove_low_variance_features(df_processed)
        
        # Remover características correlacionadas
        if remove_correlated:
            df_processed = self.remove_highly_correlated_features(df_processed)
        
        # Escalar características
        if scale_features:
            df_processed = self.scale_features(df_processed)
        
        logger.info("Pipeline de preprocesamiento completado")
        return df_processed
    
    def save_preprocessor(self, path: str):
        """
        Guardar el preprocesador
        
        Args:
            path: Ruta donde guardar el preprocesador
        """
        import joblib
        
        preprocessor_data = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'outlier_bounds': self.outlier_bounds,
            'config': self.config
        }
        
        joblib.dump(preprocessor_data, path)
        logger.info(f"Preprocesador guardado en: {path}")
    
    def load_preprocessor(self, path: str):
        """
        Cargar preprocesador guardado
        
        Args:
            path: Ruta del preprocesador
        """
        import joblib
        
        preprocessor_data = joblib.load(path)
        
        self.scalers = preprocessor_data['scalers']
        self.imputers = preprocessor_data['imputers']
        self.outlier_bounds = preprocessor_data['outlier_bounds']
        self.config = preprocessor_data['config']
        
        logger.info(f"Preprocesador cargado desde: {path}")
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Obtener resumen del preprocesamiento aplicado
        
        Returns:
            Diccionario con resumen de operaciones
        """
        summary = {
            'scalers_fitted': list(self.scalers.keys()),
            'imputers_fitted': list(self.imputers.keys()),
            'outlier_detection_applied': len(self.outlier_bounds) > 0,
            'outlier_bounds_set': len(self.outlier_bounds),
            'config_loaded': bool(self.config)
        }
        
        return summary