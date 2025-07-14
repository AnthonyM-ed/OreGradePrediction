import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)

class GeologicalFeatureEngineer:
    """Ingeniero de características para datos geológicos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear características espaciales
        
        Args:
            df: DataFrame con columnas latitude, longitude
            
        Returns:
            DataFrame con características espaciales añadidas
        """
        df_features = df.copy()
        
        # Coordenadas normalizadas
        df_features['lat_norm'] = (df_features['latitude'] - df_features['latitude'].mean()) / df_features['latitude'].std()
        df_features['lon_norm'] = (df_features['longitude'] - df_features['longitude'].mean()) / df_features['longitude'].std()
        
        # Distancia desde el centro del depósito
        center_lat = df_features['latitude'].mean()
        center_lon = df_features['longitude'].mean()
        
        df_features['distance_to_center'] = np.sqrt(
            (df_features['latitude'] - center_lat)**2 + 
            (df_features['longitude'] - center_lon)**2
        )
        
        # Características angulares
        df_features['angle_from_center'] = np.arctan2(
            df_features['latitude'] - center_lat,
            df_features['longitude'] - center_lon
        )
        
        # Cuadrantes geográficos
        df_features['quadrant'] = (
            (df_features['latitude'] > center_lat).astype(int) * 2 + 
            (df_features['longitude'] > center_lon).astype(int)
        )
        
        logger.info("Características espaciales creadas")
        return df_features
    
    def create_neighbor_features(self, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """
        Crear características basadas en vecinos más cercanos
        
        Args:
            df: DataFrame con coordenadas y valores
            k: Número de vecinos a considerar
            
        Returns:
            DataFrame con características de vecinos
        """
        df_features = df.copy()
        
        # Preparar coordenadas
        coordinates = df_features[['latitude', 'longitude']].values
        
        # Encontrar vecinos más cercanos
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)
        
        # Excluir el punto mismo (primer vecino)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Características de vecinos
        df_features['avg_neighbor_distance'] = distances.mean(axis=1)
        df_features['min_neighbor_distance'] = distances.min(axis=1)
        df_features['max_neighbor_distance'] = distances.max(axis=1)
        
        # Promedio de valores de vecinos
        neighbor_grades = []
        for i, neighbor_idx in enumerate(indices):
            neighbor_values = df_features.iloc[neighbor_idx]['standardized_grade_ppm'].values
            neighbor_grades.append(neighbor_values.mean())
        
        df_features['avg_neighbor_grade'] = neighbor_grades
        
        # Diferencia con promedio de vecinos
        df_features['grade_vs_neighbors'] = (
            df_features['standardized_grade_ppm'] - df_features['avg_neighbor_grade']
        )
        
        logger.info(f"Características de {k} vecinos más cercanos creadas")
        return df_features
    
    def create_statistical_features(self, df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Crear características estadísticas locales
        
        Args:
            df: DataFrame con datos
            window_size: Tamaño de ventana para estadísticas móviles
            
        Returns:
            DataFrame con características estadísticas
        """
        df_features = df.copy()
        
        # Ordenar por coordenadas para crear ventanas espaciales
        df_features = df_features.sort_values(['latitude', 'longitude'])
        
        # Estadísticas móviles
        df_features['grade_rolling_mean'] = (
            df_features['standardized_grade_ppm'].rolling(window=window_size, min_periods=1).mean()
        )
        df_features['grade_rolling_std'] = (
            df_features['standardized_grade_ppm'].rolling(window=window_size, min_periods=1).std()
        )
        df_features['grade_rolling_median'] = (
            df_features['standardized_grade_ppm'].rolling(window=window_size, min_periods=1).median()
        )
        
        # Características de variabilidad local
        df_features['local_coefficient_variation'] = (
            df_features['grade_rolling_std'] / df_features['grade_rolling_mean']
        )
        
        # Percentiles locales
        df_features['grade_rolling_q25'] = (
            df_features['standardized_grade_ppm'].rolling(window=window_size, min_periods=1).quantile(0.25)
        )
        df_features['grade_rolling_q75'] = (
            df_features['standardized_grade_ppm'].rolling(window=window_size, min_periods=1).quantile(0.75)
        )
        
        # Rango intercuartil local
        df_features['local_iqr'] = (
            df_features['grade_rolling_q75'] - df_features['grade_rolling_q25']
        )
        
        logger.info("Características estadísticas creadas")
        return df_features
    
    def create_geological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear características geológicas específicas
        
        Args:
            df: DataFrame con datos geológicos
            
        Returns:
            DataFrame con características geológicas
        """
        df_features = df.copy()
        
        # Codificar variables categóricas
        if 'DataSet' in df_features.columns:
            if 'DataSet' not in self.label_encoders:
                self.label_encoders['DataSet'] = LabelEncoder()
                df_features['dataset_encoded'] = self.label_encoders['DataSet'].fit_transform(df_features['DataSet'])
            else:
                df_features['dataset_encoded'] = self.label_encoders['DataSet'].transform(df_features['DataSet'])
        
        # Características de concentración
        df_features['grade_log'] = np.log1p(df_features['standardized_grade_ppm'])  # log(1 + x) para evitar log(0)
        df_features['grade_sqrt'] = np.sqrt(df_features['standardized_grade_ppm'])
        df_features['grade_squared'] = df_features['standardized_grade_ppm'] ** 2
        
        # Clasificación de grados
        grade_quantiles = df_features['standardized_grade_ppm'].quantile([0.25, 0.5, 0.75])
        df_features['grade_category'] = pd.cut(
            df_features['standardized_grade_ppm'],
            bins=[-np.inf, grade_quantiles[0.25], grade_quantiles[0.5], grade_quantiles[0.75], np.inf],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Codificar categorías de grado
        if 'grade_category' not in self.label_encoders:
            self.label_encoders['grade_category'] = LabelEncoder()
            df_features['grade_category_encoded'] = self.label_encoders['grade_category'].fit_transform(df_features['grade_category'])
        
        # Características de anomalías
        mean_grade = df_features['standardized_grade_ppm'].mean()
        std_grade = df_features['standardized_grade_ppm'].std()
        
        df_features['is_anomaly'] = (
            np.abs(df_features['standardized_grade_ppm'] - mean_grade) > 2 * std_grade
        ).astype(int)
        
        df_features['anomaly_score'] = np.abs(df_features['standardized_grade_ppm'] - mean_grade) / std_grade
        
        logger.info("Características geológicas creadas")
        return df_features
    
    def create_multi_element_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear características para análisis multi-elemento
        
        Args:
            df: DataFrame con múltiples elementos
            
        Returns:
            DataFrame con características multi-elemento
        """
        if 'Element' not in df.columns:
            logger.warning("No hay columna 'Element' para crear características multi-elemento")
            return df
        
        df_features = df.copy()
        
        # Pivot para tener elementos como columnas
        df_pivot = df_features.pivot_table(
            index=['Hole_ID', 'latitude', 'longitude'],
            columns='Element',
            values='standardized_grade_ppm',
            aggfunc='mean'
        ).reset_index()
        
        # Llenar valores faltantes con 0
        df_pivot = df_pivot.fillna(0)
        
        # Crear ratios entre elementos (si existen)
        elements = [col for col in df_pivot.columns if col not in ['Hole_ID', 'latitude', 'longitude']]
        
        if len(elements) > 1:
            for i, elem1 in enumerate(elements):
                for elem2 in elements[i+1:]:
                    # Ratio entre elementos
                    df_pivot[f'{elem1}_{elem2}_ratio'] = (
                        df_pivot[elem1] / (df_pivot[elem2] + 1e-8)  # Evitar división por cero
                    )
                    
                    # Suma de elementos
                    df_pivot[f'{elem1}_{elem2}_sum'] = df_pivot[elem1] + df_pivot[elem2]
        
        # Concentración total
        df_pivot['total_concentration'] = df_pivot[elements].sum(axis=1)
        
        # Diversidad de elementos (número de elementos con concentración > 0)
        df_pivot['element_diversity'] = (df_pivot[elements] > 0).sum(axis=1)
        
        logger.info("Características multi-elemento creadas")
        return df_pivot
    
    def create_all_features(self, df: pd.DataFrame, include_neighbors: bool = True) -> pd.DataFrame:
        """
        Crear todas las características disponibles
        
        Args:
            df: DataFrame con datos originales
            include_neighbors: Si incluir características de vecinos (computacionalmente costoso)
            
        Returns:
            DataFrame con todas las características
        """
        logger.info("Iniciando creación de características completas")
        
        # Crear características básicas
        df_features = self.create_spatial_features(df)
        df_features = self.create_statistical_features(df_features)
        df_features = self.create_geological_features(df_features)
        
        # Crear características de vecinos si se solicita
        if include_neighbors and len(df_features) > 10:  # Solo si hay suficientes datos
            df_features = self.create_neighbor_features(df_features)
        
        # Guardar nombres de características
        self.feature_names = [col for col in df_features.columns if col not in [
            'Hole_ID', 'Element', 'DataSet', 'standardized_grade_ppm', 'latitude', 'longitude'
        ]]
        
        logger.info(f"Creadas {len(self.feature_names)} características")
        return df_features
    
    def get_feature_importance_names(self) -> List[str]:
        """Obtener nombres de características para importancia"""
        return self.feature_names.copy()
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transformar nuevos datos usando los mismos encoders
        
        Args:
            df: DataFrame con nuevos datos
            
        Returns:
            DataFrame transformado
        """
        return self.create_all_features(df, include_neighbors=False)