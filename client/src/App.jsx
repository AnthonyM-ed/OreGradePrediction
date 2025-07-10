import { useEffect, useState } from 'react';
import Sidebar from './components/Sidebar/Sidebar';
import HeatMapLibre from './components/HeatMapLibre/HeatMapLibre';
import IAChat from './components/IAChat/IAChat';
import './App.css';

function App() {
  const [data, setData] = useState([]); // Datos de la tabla seleccionada
  const [tables, setTables] = useState([]); // Listado de tablas
  const [sidebarStructure, setSidebarStructure] = useState([]); // Estructura del sidebar
  const [selectedTable, setSelectedTable] = useState(null); // Tabla seleccionada
  const [collapsed, setCollapsed] = useState(false); // Estado del sidebar (colapsado/expandido)

  // Cargar configuración y tablas desde el backend
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/config')
      .then((res) => res.json())
      .then((config) => {
        setSidebarStructure(config.sidebarStructure);
        setTables(config.validTables);

        // Si no hay tablas disponibles, mostramos un mensaje o imagen
        if (config.validTables.length === 0 || selectedTable === null) {
          setSelectedTable(null); // No seleccionamos ninguna tabla si no hay disponibles
        }
      })
      .catch((err) => console.error('Error al cargar configuración:', err));
  }, []);

  // Cargar datos de la tabla seleccionada
  useEffect(() => {
  if (!selectedTable || selectedTable === 'HeatMap') return; // ← evita cargar datos si es HeatMap

  fetch(`http://127.0.0.1:8000/api/table/${selectedTable}`)
    .then((res) => res.json())
    .then((info) => setData(info))
    .catch((err) => console.error('Error al obtener datos:', err.message));
}, [selectedTable]);

  // Manejar la selección de tabla
  const handleTableSelect = (table) => {
    setSelectedTable(table);
  };

  // Manejar el cambio de estado del sidebar (colapsar/expandir)
  const handleCollapseToggle = () => {
    setCollapsed((prev) => !prev);
  };


  return (
    <div style={{ display: 'flex' }}>
      <Sidebar
        sidebarStructure={sidebarStructure}
        tables={tables}
        onTableSelect={handleTableSelect}
        onCollapseToggle={handleCollapseToggle}
        selectedTable={selectedTable} // Pasamos la tabla seleccionada al Sidebar
        collapsed={collapsed} 
      />
      <div
        style={{
          flex: 1,
          padding: '1px',
          marginLeft: collapsed ? '64px' : '265px',
          transition: 'margin-left 0.3s ease-in-out',
        }}
      >
        {selectedTable === "HeatMap" ? (
          <>
            <h1 style={{ marginLeft: '16px' }}>Visualización de Mapa de Calor</h1>
            <HeatMapLibre />
          </>
        ) : (
          <>
            <h1 style={{ marginLeft: '16px' }}>
              Datos de la tabla: {selectedTable || 'Selecciona una tabla'}
            </h1>

            {selectedTable === null ? (
              <div className="no-tables-message">
                <p>No hay tablas seleccionadas</p>
                <img src="/path/to/no-tables-image.png" alt="No Tables" />
              </div>
            ) : tables.length === 0 ? (
              <div>No hay tablas disponibles.</div>
            ) : data.length === 0 ? (
              <div>Cargando...</div>
            ) : (
              <table className="styled-table">
                <thead>
                  <tr>
                    {Object.keys(data[0]).length > 0 &&
                      Object.keys(data[0]).map((header, index) => (
                        <th key={index}>{header}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {data.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {Object.values(row).map((value, colIndex) => (
                        <td key={colIndex}>
                          {typeof value === 'object' && value !== null
                            ? JSON.stringify(value)
                            : value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </div>
      <IAChat />
    </div>
  );
}

export default App;
