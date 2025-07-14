"""
Database Connection Management
==============================

Handles SQL Server connections and connection pooling for geological data access.
"""

import logging
import pyodbc
from typing import Dict, Any, Optional
from contextlib import contextmanager
from django.conf import settings

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages SQL Server database connections"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or getattr(settings, 'MSSQL_CONFIG', {})
        self._connection = None
        
    def get_connection_string(self) -> str:
        """Build SQL Server connection string"""
        server = self.config.get('server')
        database = self.config.get('database')
        username = self.config.get('username')
        password = self.config.get('password')
        port = self.config.get('port', '1433')
        driver = self.config.get('driver', 'ODBC Driver 17 for SQL Server')
        
        if not all([server, database, username, password]):
            raise ValueError("Missing required database configuration")
        
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server},{port};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
            f"Encrypt=no;"
        )
        
        return conn_str
    
    def connect(self) -> pyodbc.Connection:
        """Establish database connection"""
        try:
            if self._connection is None or self._connection.closed:
                conn_str = self.get_connection_string()
                self._connection = pyodbc.connect(conn_str)
                logger.info("Database connection established")
            return self._connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            logger.info("Database connection closed")
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def __enter__(self):
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

@contextmanager
def get_db_connection(config: Dict[str, Any] = None):
    """Context manager for database connections"""
    db_conn = DatabaseConnection(config)
    try:
        conn = db_conn.connect()
        yield conn
    finally:
        db_conn.close()

def get_sql_server_connection() -> pyodbc.Connection:
    """Get SQL Server connection (legacy function for compatibility)"""
    db_conn = DatabaseConnection()
    return db_conn.connect()

class ConnectionPool:
    """Simple connection pool for database connections"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.connections = []
        self.config = getattr(settings, 'MSSQL_CONFIG', {})
    
    def get_connection(self) -> pyodbc.Connection:
        """Get connection from pool or create new one"""
        # Simple implementation - can be enhanced with proper pooling
        if self.connections:
            conn = self.connections.pop()
            if not conn.closed:
                return conn
        
        # Create new connection
        db_conn = DatabaseConnection(self.config)
        return db_conn.connect()
    
    def return_connection(self, conn: pyodbc.Connection):
        """Return connection to pool"""
        if not conn.closed and len(self.connections) < self.max_connections:
            self.connections.append(conn)
        else:
            conn.close()
    
    def close_all(self):
        """Close all connections in pool"""
        for conn in self.connections:
            if not conn.closed:
                conn.close()
        self.connections.clear()

# Global connection pool instance
connection_pool = ConnectionPool()
