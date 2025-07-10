import json
import os
import pyodbc
from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from openai import OpenAI
import logging
import httpx

logger = logging.getLogger(__name__)

# Inicializar OpenAI
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    http_client=httpx.Client(proxy=None),  # 🔒 asegura que no se use proxies
)

def get_sql_server_connection():
    """Crear conexión directa a SQL Server"""
    try:
        config = settings.MSSQL_CONFIG
        connection_string = (
            f"DRIVER={{{config['driver']}}};"
            f"SERVER={config['server']},{config['port']};"
            f"DATABASE={config['database']};"
            f"UID={config['username']};"
            f"PWD={config['password']};"
            f"TrustServerCertificate={config['trust_server_certificate']};"
            f"Encrypt={config['encrypt']};"
        )
        return pyodbc.connect(connection_string)
    except Exception as e:
        logger.error(f"Error conectando a SQL Server: {e}")
        raise

def load_config():
    """Cargar configuración desde tables.json"""
    try:
        config_path = os.path.join(settings.BASE_DIR, 'config', 'tables.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return {"validTables": []}

@api_view(['GET'])
def get_config(request):
    """Endpoint para obtener la configuración"""
    config_data = load_config()
    return Response(config_data)

@api_view(['GET'])
def get_table_data(request, table_name):
    """Endpoint para obtener datos de una tabla específica"""
    config_data = load_config()
    
    # Validar tabla
    if table_name not in config_data.get('validTables', []):
        return Response(
            {'message': 'Tabla o vista no válida'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        conn = get_sql_server_connection()
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT TOP 5 * FROM {table_name}")
        
        # Obtener nombres de columnas
        columns = [column[0] for column in cursor.description]
        
        # Obtener datos
        rows = cursor.fetchall()
        
        # Convertir a lista de diccionarios
        result = []
        for row in rows:
            result.append(dict(zip(columns, row)))
        
        cursor.close()
        conn.close()
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Error ejecutando consulta SQL: {e}")
        
        if 'Invalid object name' in str(e):
            return Response(
                {'message': 'Vista no encontrada'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response(
            {'message': 'Error al conectar a la base de datos'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_heatmap(request):
    """Endpoint para obtener datos del heatmap en formato GeoJSON"""
    element = request.GET.get('element', 'Cu')  # Default Cu si no viene
    
    try:
        conn = get_sql_server_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                C.LL_Long as longitude,
                C.LL_Lat as latitude,
                G.weighted_grade as concentration
            FROM dbo.tblHoleGrades_MapData_Standardized_Cache G
            JOIN dbo.tblDHColl C ON G.Hole_ID = C.Hole_ID
            WHERE G.Element = ?
                AND C.LL_Lat IS NOT NULL
                AND C.LL_Long IS NOT NULL
        """
        
        cursor.execute(query, element)
        
        # Obtener nombres de columnas
        columns = [column[0] for column in cursor.description]
        
        # Obtener datos
        rows = cursor.fetchall()
        
        # Construir GeoJSON
        features = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            features.append({
                'type': 'Feature',
                'properties': {
                    'concentration': row_dict['concentration']
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row_dict['longitude'], row_dict['latitude']]
                }
            })
        
        cursor.close()
        conn.close()
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        return Response(geojson)
        
    except Exception as e:
        logger.error(f"Error en consulta heatmap: {e}")
        return Response(
            {'error': 'Error al obtener datos del heatmap'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def ask_question(request):
    """Endpoint para responder preguntas en lenguaje natural usando datos de SQL Server + OpenAI"""
    question = request.data.get('question')

    if not question:
        return Response({'error': 'Falta la pregunta.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Simplificación: asumimos que preguntan por un elemento como 'Au'
        conn = get_sql_server_connection()
        cursor = conn.cursor()

        # Extraer el elemento (esto puedes mejorar con NLP si lo deseas)
        element = None
        for symbol in ['Au', 'Cu', 'Zn', 'Pb', 'Fe', 'Ag']:
            if symbol.lower() in question.lower():
                element = symbol
                break

        if not element:
            return Response({'error': 'Elemento no identificado en la pregunta.'}, status=status.HTTP_400_BAD_REQUEST)

        # Consulta SQL: mayor concentración del elemento
        query = """
            SELECT TOP 1 
                C.LL_Lat as latitude,
                C.LL_Long as longitude,
                G.weighted_grade as concentration
            FROM dbo.tblHoleGrades_MapData_Standardized_Cache G
            JOIN dbo.tblDHColl C ON G.Hole_ID = C.Hole_ID
            WHERE G.Element = ?
              AND C.LL_Lat IS NOT NULL
              AND C.LL_Long IS NOT NULL
            ORDER BY G.weighted_grade DESC
        """

        cursor.execute(query, (element,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return Response({'error': 'No se encontraron datos para ese elemento.'}, status=status.HTTP_404_NOT_FOUND)

        lat, lon, grade = row

        # Generar respuesta en lenguaje natural con OpenAI
        prompt = (
            f"Tengo el siguiente dato:\n"
            f"Elemento: {element}\n"
            f"Latitud: {lat}\n"
            f"Longitud: {lon}\n"
            f"Concentración: {grade}\n\n"
            f"Responde en lenguaje natural dónde se encuentra la mayor concentración de {element}. "
            f"No muestres el SQL ni código. Sé claro y conciso."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente geológico."},
                {"role": "user", "content": prompt}
            ],
        )

        answer = response.choices[0].message.content.strip()
        print("Respuesta generada por OpenAI:", answer)

        return Response({
            "question": question,
            "element": element,
            "location": {"lat": lat, "lon": lon},
            "concentration": grade,
            "answer": answer
        })
    

    except Exception as e:
        logger.error(f"Error en consulta natural: {e}")
        return Response({'error': 'Error al procesar la pregunta', 'details': str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)