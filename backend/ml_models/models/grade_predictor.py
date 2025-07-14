import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from django.conf import settings

logger = logging.getLogger(__name__)

class GradePredictor:
    """Predictor de ley de minerales usando XGBoost"""
    
    def __init__(self, config_path: str = None):
        """
        Inicializar el predictor
        
        Args:
            config_path: Ruta al archivo de configuración ML
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_variable = 'weighted_grade'
        self.is_trained = False
        self.model_metadata = {}
        
        # Cargar configuración
        if config_path is None:
            config_path = os.path.join(settings.BASE_DIR, 'config', 'ml_config.json')
        
        self.config = self._load_config(config_path)
        self.model_params = self.config.get('xgboost_params', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Cargar configuración desde archivo JSON"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuración por defecto"""
        return {
            "model_config": {
                "algorithm": "xgboost",
                "target_variable": "weighted_grade",
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5
            },
            "xgboost_params": {
                "objective": "reg:squarederror",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            },
            "validation_metrics": {
                "primary": ["r2", "mae", "rmse"],
                "secondary": ["mape", "explained_variance"]
            }
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preparar características para el modelo
        
        Args:
            df: DataFrame con datos procesados
            
        Returns:
            Tuple con DataFrame de características y lista de nombres
        """
        # Excluir columnas que no son características
        exclude_columns = [
            'Hole_ID', 'Element', 'DataSet', 'weighted_grade', 
            'latitude', 'longitude', 'grade_category'
        ]
        
        # Seleccionar características numéricas
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Filtrar solo columnas numéricas
        numeric_features = []
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                numeric_features.append(col)
        
        X = df[numeric_features].copy()
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        
        # Remover características con varianza cero
        variance_threshold = 1e-8
        low_variance_cols = X.columns[X.var() < variance_threshold]
        if len(low_variance_cols) > 0:
            logger.warning(f"Removiendo {len(low_variance_cols)} características con baja varianza")
            X = X.drop(columns=low_variance_cols)
        
        self.feature_names = list(X.columns)
        logger.info(f"Preparadas {len(self.feature_names)} características")
        
        return X, self.feature_names
    
    def train(self, df: pd.DataFrame, validation_split: bool = True) -> Dict[str, Any]:
        """
        Entrenar el modelo
        
        Args:
            df: DataFrame con datos de entrenamiento
            validation_split: Si dividir en entrenamiento/validación
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo")
        
        # Preparar características
        X, feature_names = self.prepare_features(df)
        y = df[self.target_variable].copy()
        
        # Verificar que tenemos datos suficientes
        if len(X) < 10:
            raise ValueError("Datos insuficientes para entrenar el modelo")
        
        # Dividir en entrenamiento y validación
        model_config = self.config.get('model_config', {})
        test_size = model_config.get('test_size', 0.2)
        random_state = model_config.get('random_state', 42)
        
        if validation_split and len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Crear y entrenar modelo XGBoost
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(X_train_scaled, y_train)
        
        # Hacer predicciones
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calcular métricas
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # Validación cruzada
        if len(X) > 20:
            cv_scores = self._cross_validate(X, y)
            metrics['cross_validation'] = cv_scores
        
        # Importancia de características
        feature_importance = self._get_feature_importance()
        metrics['feature_importance'] = feature_importance
        
        # Metadata del modelo
        self.model_metadata = {
            'model_type': 'XGBoost',
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'target_variable': self.target_variable,
            'model_params': self.model_params,
            'metrics': metrics
        }
        
        self.is_trained = True
        logger.info("Modelo entrenado exitosamente")
        
        return metrics
    
    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, Any]:
        """Calcular métricas de evaluación"""
        
        def calculate_mape(y_true, y_pred):
            """Calcular MAPE evitando división por cero"""
            mask = y_true != 0
            if mask.sum() == 0:
                return np.inf
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        metrics = {
            'train': {
                'r2': float(r2_score(y_train, y_pred_train)),
                'mae': float(mean_absolute_error(y_train, y_pred_train)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mape': float(calculate_mape(y_train, y_pred_train)),
                'explained_variance': float(explained_variance_score(y_train, y_pred_train))
            },
            'test': {
                'r2': float(r2_score(y_test, y_pred_test)),
                'mae': float(mean_absolute_error(y_test, y_pred_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mape': float(calculate_mape(y_test, y_pred_test)),
                'explained_variance': float(explained_variance_score(y_test, y_pred_test))
            }
        }
        
        return metrics
    
    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Realizar validación cruzada"""
        try:
            cv_folds = self.config.get('model_config', {}).get('cv_folds', 5)
            
            # Escalar datos para validación cruzada
            X_scaled = self.scaler.fit_transform(X)
            
            # Crear modelo temporal para CV
            cv_model = xgb.XGBRegressor(**self.model_params)
            
            # Validación cruzada con diferentes métricas
            cv_r2 = cross_val_score(cv_model, X_scaled, y, cv=cv_folds, scoring='r2')
            cv_mae = cross_val_score(cv_model, X_scaled, y, cv=cv_folds, scoring='neg_mean_absolute_error')
            cv_rmse = cross_val_score(cv_model, X_scaled, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
            
            cv_results = {
                'r2_scores': cv_r2.tolist(),
                'r2_mean': float(cv_r2.mean()),
                'r2_std': float(cv_r2.std()),
                'mae_scores': (-cv_mae).tolist(),
                'mae_mean': float(-cv_mae.mean()),
                'mae_std': float(cv_mae.std()),
                'rmse_scores': (-cv_rmse).tolist(),
                'rmse_mean': float(-cv_rmse.mean()),
                'rmse_std': float(cv_rmse.std())
            }
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error en validación cruzada: {e}")
            return {}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Obtener importancia de características"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Ordenar por importancia
        sorted_importance = dict(sorted(feature_importance.items(), 
                                     key=lambda item: item[1], reverse=True))
        
        return sorted_importance
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Hacer predicciones
        
        Args:
            df: DataFrame con datos a predecir
            
        Returns:
            Array con predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Preparar características
        X, _ = self.prepare_features(df)
        
        # Asegurar que tenemos las mismas características
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Características faltantes: {missing_features}")
            for feature in missing_features:
                X[feature] = 0
        
        # Reordenar columnas
        X = X[self.feature_names]
        
        # Escalar y predecir
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hacer predicciones con intervalos de confianza
        
        Args:
            df: DataFrame con datos a predecir
            
        Returns:
            Tuple con predicciones y desviación estándar estimada
        """
        predictions = self.predict(df)
        
        # Estimar incertidumbre basada en métricas de entrenamiento
        if 'test' in self.model_metadata.get('metrics', {}):
            rmse = self.model_metadata['metrics']['test']['rmse']
            uncertainty = np.full_like(predictions, rmse)
        else:
            uncertainty = np.full_like(predictions, 0.1)
        
        return predictions, uncertainty
    
    def save_model(self, model_path: str = None, scaler_path: str = None, 
                   metadata_path: str = None):
        """
        Guardar el modelo entrenado
        
        Args:
            model_path: Ruta para guardar el modelo
            scaler_path: Ruta para guardar el scaler
            metadata_path: Ruta para guardar metadata
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Rutas por defecto
        data_dir = os.path.join(settings.BASE_DIR, 'data', 'models')
        os.makedirs(data_dir, exist_ok=True)
        
        if model_path is None:
            model_path = os.path.join(data_dir, 'grade_model_v1.joblib')
        if scaler_path is None:
            scaler_path = os.path.join(data_dir, 'scaler.joblib')
        if metadata_path is None:
            metadata_path = os.path.join(data_dir, 'model_metadata.json')
        
        # Guardar modelo y scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Guardar metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info(f"Modelo guardado en: {model_path}")
        logger.info(f"Scaler guardado en: {scaler_path}")
        logger.info(f"Metadata guardada en: {metadata_path}")
    
    def load_model(self, model_path: str = None, scaler_path: str = None,
                   metadata_path: str = None):
        """
        Cargar modelo entrenado
        
        Args:
            model_path: Ruta del modelo
            scaler_path: Ruta del scaler
            metadata_path: Ruta de metadata
        """
        # Rutas por defecto
        data_dir = os.path.join(settings.BASE_DIR, 'data', 'models')
        
        if model_path is None:
            model_path = os.path.join(data_dir, 'grade_model_v1.joblib')
        if scaler_path is None:
            scaler_path = os.path.join(data_dir, 'scaler.joblib')
        if metadata_path is None:
            metadata_path = os.path.join(data_dir, 'model_metadata.json')
        
        try:
            # Cargar modelo y scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Cargar metadata
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            self.feature_names = self.model_metadata.get('feature_names', [])
            self.is_trained = True
            
            logger.info("Modelo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Obtener resumen del modelo"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "model_type": self.model_metadata.get('model_type', 'XGBoost'),
            "n_features": self.model_metadata.get('n_features', 0),
            "training_samples": self.model_metadata.get('training_samples', 0),
            "validation_samples": self.model_metadata.get('validation_samples', 0),
            "metrics": self.model_metadata.get('metrics', {}),
            "feature_names": self.feature_names[:10]  # Primeras 10 características
        }