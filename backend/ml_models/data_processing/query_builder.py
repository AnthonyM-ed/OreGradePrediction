import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Tipos de consultas SQL disponibles"""
    DRILLING_DATA = "drilling_data"
    MULTI_ELEMENT = "multi_element"
    SPATIAL_CONTEXT = "spatial_context"
    SUMMARY_STATS = "summary_stats"
    QUALITY_CHECK = "quality_check"
    COORDINATES = "coordinates"

@dataclass
class QueryFilter:
    """Filtro para consultas SQL"""
    column: str
    operator: str  # =, >, <, >=, <=, IN, BETWEEN, LIKE
    value: Any
    table_alias: Optional[str] = None

class SQLQueryBuilder:
    """Constructor de consultas SQL optimizadas para datos geológicos"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Use the new unified view instead of separate tables
        self.main_view = self.config.get('main_view', 'vw_HoleSamples_ElementGrades')
        # Keep for backward compatibility, but prefer the unified view
        self.main_table = self.config.get('main_table', 'vw_HoleSamples_ElementGrades')
        self.coord_table = self.config.get('coordinates_table', 'tblDHColl')  # Not used with new view
        
    def build_drilling_query(self, elements: List[str] = None, 
                           filters: List[QueryFilter] = None,
                           limit: Optional[int] = None) -> Tuple[str, List[Any]]:
        """
        Construir consulta para datos de perforación
        
        Args:
            elements: Lista de elementos químicos
            filters: Filtros adicionales
            limit: Límite de registros
            
        Returns:
            Tuple con query SQL y parámetros
        """
        if elements is None:
            elements = ['Cu']
        
        # Construir cláusula WHERE para elementos
        if len(elements) == 1:
            element_clause = "Element = ?"
            params = [elements[0]]
        else:
            placeholders = ', '.join(['?' for _ in elements])
            element_clause = f"Element IN ({placeholders})"
            params = elements
        
        # Query base using the new unified view
        query = f"""
            SELECT 
                Hole_ID,
                Element,
                DataSet,
                standardized_grade_ppm,
                latitude,
                longitude,
                elevation,
                Depth_From as depth_from,
                Depth_To as depth_to,
                Interval_Length as interval_length,
                SampleID,
                LabCode
            FROM dbo.{self.main_view}
            WHERE {element_clause}
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
                AND standardized_grade_ppm IS NOT NULL
                AND standardized_grade_ppm > 0
        """
        
        # Agregar filtros adicionales
        if filters:
            for filter_obj in filters:
                # Remove table aliases since we're using a single view
                column_name = filter_obj.column
                
                if filter_obj.operator == 'BETWEEN':
                    query += f" AND {column_name} BETWEEN ? AND ?"
                    params.extend([filter_obj.value[0], filter_obj.value[1]])
                elif filter_obj.operator == 'IN':
                    placeholders = ', '.join(['?' for _ in filter_obj.value])
                    query += f" AND {column_name} IN ({placeholders})"
                    params.extend(filter_obj.value)
                else:
                    query += f" AND {column_name} {filter_obj.operator} ?"
                    params.append(filter_obj.value)
        
        # Ordenar por coordenadas para análisis espacial
        query += " ORDER BY latitude, longitude, Hole_ID"
        
        # Agregar límite si se especifica
        if limit:
            query += f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
        
        logger.info(f"Query construida para {len(elements)} elementos con {len(filters or [])} filtros")
        return query, params
    
    def build_spatial_query(self, center_lat: float, center_lon: float,
                          radius_km: float = 10.0, elements: List[str] = None) -> Tuple[str, List[Any]]:
        """
        Construir consulta espacial con cálculo de distancia
        
        Args:
            center_lat: Latitud central
            center_lon: Longitud central
            radius_km: Radio en kilómetros
            elements: Lista de elementos
            
        Returns:
            Tuple con query SQL y parámetros
        """
        if elements is None:
            elements = ['Cu']
        
        # Conversión aproximada de km a grados
        radius_deg = radius_km / 111.0
        
        # Construir cláusula para elementos
        if len(elements) == 1:
            element_clause = "Element = ?"
            params = [elements[0]]
        else:
            placeholders = ', '.join(['?' for _ in elements])
            element_clause = f"Element IN ({placeholders})"
            params = elements
        
        query = f"""
            SELECT 
                Hole_ID,
                Element,
                DataSet,
                standardized_grade_ppm,
                latitude,
                longitude,
                elevation,
                -- Cálculo de distancia usando fórmula haversine simplificada
                SQRT(POWER((latitude - ?) * 111.0, 2) + 
                     POWER((longitude - ?) * 111.0 * COS(RADIANS((latitude + ?) / 2)), 2)) as distance_km,
                -- Ángulo desde el centro
                ATAN2(latitude - ?, longitude - ?) as angle_radians
            FROM dbo.{self.main_view}
            WHERE {element_clause}
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
                AND standardized_grade_ppm IS NOT NULL
                AND standardized_grade_ppm > 0
                AND SQRT(POWER(latitude - ?, 2) + POWER(longitude - ?, 2)) <= ?
            ORDER BY distance_km, standardized_grade_ppm DESC
        """
        
        # Agregar parámetros para cálculos espaciales
        spatial_params = [center_lat, center_lon, center_lat, center_lat, center_lon, 
                         center_lat, center_lon, radius_deg]
        params.extend(spatial_params)
        
        logger.info(f"Query espacial construida para radio de {radius_km} km")
        return query, params
    
    def build_summary_query(self, elements: List[str] = None,
                          group_by: str = "Element") -> Tuple[str, List[Any]]:
        """
        Construir consulta de resumen estadístico
        
        Args:
            elements: Lista de elementos
            group_by: Campo para agrupar (Element, DataSet, etc.)
            
        Returns:
            Tuple con query SQL y parámetros
        """
        if elements is None:
            elements = ['Cu']
        
        # Construir cláusula para elementos
        if len(elements) == 1:
            element_clause = "Element = ?"
            params = [elements[0]]
        else:
            placeholders = ', '.join(['?' for _ in elements])
            element_clause = f"Element IN ({placeholders})"
            params = elements
        
        valid_group_fields = {
            'Element': 'Element',
            'DataSet': 'DataSet',
            'LabCode': 'LabCode',
            'Elevation': 'CASE WHEN elevation < 1000 THEN \'Low\' WHEN elevation < 2000 THEN \'Medium\' ELSE \'High\' END'
        }
        
        group_field = valid_group_fields.get(group_by, 'Element')
        
        query = f"""
            SELECT 
                {group_field} as group_field,
                COUNT(*) as record_count,
                COUNT(DISTINCT Hole_ID) as unique_holes,
                AVG(standardized_grade_ppm) as avg_grade,
                MIN(standardized_grade_ppm) as min_grade,
                MAX(standardized_grade_ppm) as max_grade,
                STDEV(standardized_grade_ppm) as std_grade,
                MIN(latitude) as min_lat,
                MAX(latitude) as max_lat,
                MIN(longitude) as min_lon,
                MAX(longitude) as max_lon,
                AVG(elevation) as avg_elevation
            FROM dbo.{self.main_view}
            WHERE {element_clause}
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
                AND standardized_grade_ppm IS NOT NULL
                AND standardized_grade_ppm > 0
            GROUP BY {group_field}
            ORDER BY avg_grade DESC
        """
        
        logger.info(f"Query de resumen construida agrupando por {group_by}")
        return query, params
    
    def build_quality_check_query(self) -> Tuple[str, List[Any]]:
        """
        Construir consulta para verificación de calidad de datos
        
        Returns:
            Tuple con query SQL y parámetros
        """
        query = f"""
            SELECT 
                'Total Records' as check_type,
                COUNT(*) as count_value,
                NULL as percentage
            FROM dbo.{self.main_view}
            
            UNION ALL
            
            SELECT 
                'Records with Coordinates' as check_type,
                COUNT(*) as count_value,
                CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dbo.{self.main_view}) as DECIMAL(5,2)) as percentage
            FROM dbo.{self.main_view}
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            
            UNION ALL
            
            SELECT 
                'Valid Grade Values' as check_type,
                COUNT(*) as count_value,
                CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dbo.{self.main_view}) as DECIMAL(5,2)) as percentage
            FROM dbo.{self.main_view}
            WHERE standardized_grade_ppm IS NOT NULL AND standardized_grade_ppm > 0
            
            UNION ALL
            
            SELECT 
                'Complete Records' as check_type,
                COUNT(*) as count_value,
                CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dbo.{self.main_view}) as DECIMAL(5,2)) as percentage
            FROM dbo.{self.main_view}
            WHERE latitude IS NOT NULL 
                AND longitude IS NOT NULL
                AND standardized_grade_ppm IS NOT NULL
                AND standardized_grade_ppm > 0
            
            UNION ALL
            
            SELECT 
                'Duplicate Sample_IDs' as check_type,
                COUNT(*) as count_value,
                NULL as percentage
            FROM (
                SELECT SampleID, Element, COUNT(*) as duplicates
                FROM dbo.{self.main_view}
                GROUP BY SampleID, Element
                HAVING COUNT(*) > 1
            ) duplicates
            
            UNION ALL
            
            SELECT 
                'Records with Lab Codes' as check_type,
                COUNT(*) as count_value,
                CAST(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dbo.{self.main_view}) as DECIMAL(5,2)) as percentage
            FROM dbo.{self.main_view}
            WHERE LabCode IS NOT NULL
            
            ORDER BY check_type
        """
        
        logger.info("Query de verificación de calidad construida")
        return query, []
    
    def build_multi_element_pivot_query(self, elements: List[str],
                                      filters: List[QueryFilter] = None) -> Tuple[str, List[Any]]:
        """
        Construir consulta pivot para análisis multi-elemento
        
        Args:
            elements: Lista de elementos a incluir
            filters: Filtros adicionales
            
        Returns:
            Tuple con query SQL y parámetros
        """
        if not elements:
            raise ValueError("Se requiere al menos un elemento")
        
        # Construir cláusula para elementos
        placeholders = ', '.join(['?' for _ in elements])
        element_clause = f"Element IN ({placeholders})"
        params = elements.copy()
        
        # Crear columnas pivotadas
        pivot_columns = []
        for element in elements:
            pivot_columns.append(f"MAX(CASE WHEN Element = '{element}' THEN standardized_grade_ppm END) as {element}")
        
        pivot_select = ',\n                '.join(pivot_columns)
        
        query = f"""
            SELECT 
                base.Hole_ID,
                base.latitude,
                base.longitude,
                base.elevation,
                base.DataSet,
                {pivot_select}
            FROM (
                SELECT DISTINCT
                    Hole_ID,
                    latitude,
                    longitude,
                    elevation,
                    DataSet
                FROM dbo.{self.main_view}
                WHERE {element_clause}
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
            ) base
            LEFT JOIN dbo.{self.main_view} V ON base.Hole_ID = V.Hole_ID
            WHERE V.Element IN ({placeholders})
        """
        
        # Duplicar parámetros para el segundo IN
        params.extend(elements)
        
        # Agregar filtros adicionales
        if filters:
            for filter_obj in filters:
                column_name = filter_obj.column
                
                if filter_obj.operator == 'BETWEEN':
                    query += f" AND {column_name} BETWEEN ? AND ?"
                    params.extend([filter_obj.value[0], filter_obj.value[1]])
                elif filter_obj.operator == 'IN':
                    placeholders_filter = ', '.join(['?' for _ in filter_obj.value])
                    query += f" AND {column_name} IN ({placeholders_filter})"
                    params.extend(filter_obj.value)
                else:
                    query += f" AND {column_name} {filter_obj.operator} ?"
                    params.append(filter_obj.value)
        
        query += """
            GROUP BY base.Hole_ID, base.latitude, base.longitude, base.elevation, base.DataSet
            ORDER BY base.latitude, base.longitude
        """
        
        logger.info(f"Query pivot construida para {len(elements)} elementos")
        return query, params
    
    def build_anomaly_detection_query(self, element: str = 'Cu',
                                    std_threshold: float = 2.0) -> Tuple[str, List[Any]]:
        """
        Construir consulta para detección de anomalías
        
        Args:
            element: Elemento químico
            std_threshold: Umbral de desviaciones estándar
            
        Returns:
            Tuple con query SQL y parámetros
        """
        query = f"""
            WITH stats AS (
                SELECT 
                    AVG(standardized_grade_ppm) as mean_grade,
                    STDEV(standardized_grade_ppm) as std_grade
                FROM dbo.{self.main_view}
                WHERE Element = ?
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND standardized_grade_ppm IS NOT NULL
                    AND standardized_grade_ppm > 0
            )
            SELECT 
                V.Hole_ID,
                V.SampleID,
                V.Element,
                V.standardized_grade_ppm,
                V.latitude,
                V.longitude,
                V.DataSet,
                V.LabCode,
                stats.mean_grade,
                stats.std_grade,
                ABS(V.standardized_grade_ppm - stats.mean_grade) / stats.std_grade as z_score,
                CASE 
                    WHEN ABS(V.standardized_grade_ppm - stats.mean_grade) / stats.std_grade > ?
                    THEN 'anomaly'
                    ELSE 'normal'
                END as anomaly_flag
            FROM dbo.{self.main_view} V
            CROSS JOIN stats
            WHERE V.Element = ?
                AND V.latitude IS NOT NULL
                AND V.longitude IS NOT NULL
                AND V.standardized_grade_ppm IS NOT NULL
                AND V.standardized_grade_ppm > 0
            ORDER BY z_score DESC
        """
        
        params = [element, std_threshold, element]
        
        logger.info(f"Query de detección de anomalías construida para {element}")
        return query, params
    
    def get_query_by_type(self, query_type: QueryType, **kwargs) -> Tuple[str, List[Any]]:
        """
        Obtener query según el tipo especificado
        
        Args:
            query_type: Tipo de consulta
            **kwargs: Parámetros específicos para cada tipo
            
        Returns:
            Tuple con query SQL y parámetros
        """
        if query_type == QueryType.DRILLING_DATA:
            return self.build_drilling_query(
                elements=kwargs.get('elements'),
                filters=kwargs.get('filters'),
                limit=kwargs.get('limit')
            )
        elif query_type == QueryType.MULTI_ELEMENT:
            return self.build_multi_element_pivot_query(
                elements=kwargs.get('elements', ['Cu']),
                filters=kwargs.get('filters')
            )
        elif query_type == QueryType.SPATIAL_CONTEXT:
            return self.build_spatial_query(
                center_lat=kwargs.get('center_lat'),
                center_lon=kwargs.get('center_lon'),
                radius_km=kwargs.get('radius_km', 10.0),
                elements=kwargs.get('elements')
            )
        elif query_type == QueryType.SUMMARY_STATS:
            return self.build_summary_query(
                elements=kwargs.get('elements'),
                group_by=kwargs.get('group_by', 'Element')
            )
        elif query_type == QueryType.QUALITY_CHECK:
            return self.build_quality_check_query()
        else:
            raise ValueError(f"Tipo de consulta no soportado: {query_type}")

# Funciones de utilidad para crear filtros
def create_filter(column: str, operator: str, value: Any, table_alias: str = None) -> QueryFilter:
    """Crear un filtro de consulta"""
    return QueryFilter(column=column, operator=operator, value=value, table_alias=table_alias)

def create_grade_filter(min_grade: float = None, max_grade: float = None) -> List[QueryFilter]:
    """Crear filtros para rango de ley"""
    filters = []
    if min_grade is not None:
        filters.append(create_filter('standardized_grade_ppm', '>=', min_grade))
    if max_grade is not None:
        filters.append(create_filter('standardized_grade_ppm', '<=', max_grade))
    return filters

def create_elevation_filter(min_elevation: float = None, max_elevation: float = None) -> List[QueryFilter]:
    """Crear filtros para rango de elevación"""
    filters = []
    if min_elevation is not None:
        filters.append(create_filter('elevation', '>=', min_elevation))
    if max_elevation is not None:
        filters.append(create_filter('elevation', '<=', max_elevation))
    return filters

def create_dataset_filter(datasets: List[str]) -> QueryFilter:
    """Crear filtro para datasets específicos"""
    return create_filter('DataSet', 'IN', datasets)

def create_lab_filter(lab_codes: List[str]) -> QueryFilter:
    """Crear filtro para códigos de laboratorio específicos"""
    return create_filter('LabCode', 'IN', lab_codes)

def create_depth_filter(min_depth: float = None, max_depth: float = None) -> List[QueryFilter]:
    """Crear filtros para rango de profundidad"""
    filters = []
    if min_depth is not None:
        filters.append(create_filter('Depth_From', '>=', min_depth))
    if max_depth is not None:
        filters.append(create_filter('Depth_To', '<=', max_depth))
    return filters