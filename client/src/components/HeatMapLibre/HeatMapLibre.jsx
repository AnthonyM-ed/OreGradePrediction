import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

function HeatMapLibre() {
    const mapContainer = useRef(null);
    const mapRef = useRef(null);
    const popupRef = useRef(null);
    const [element, setElement] = useState('Cu');
    const [loading, setLoading] = useState(false);
    const cache = useRef({});

    // Rangos mínimos y máximos de concentración por elemento
    const elementRanges = {
        Cu: { min: 0, max: 8957.488006, unit: 'ppm' },
        Au: { min: 0, max: 11.904356, unit: 'ppm' },
        Zn: { min: 0, max: 79453.322416, unit: 'ppm' },
    };

    const [mapLoaded, setMapLoaded] = useState(false);

    useEffect(() => {
        mapRef.current = new maplibregl.Map({
            container: mapContainer.current,
            style: {
                version: 8,
                sources: {
                    opentopo: {
                        type: 'raster',
                        tiles: [
                            'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
                            'https://b.tile.opentopomap.org/{z}/{x}/{y}.png',
                            'https://c.tile.opentopomap.org/{z}/{x}/{y}.png',
                        ],
                        tileSize: 256,
                        attribution:
                            'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
                    },
                },
                layers: [
                    {
                        id: 'opentopo',
                        type: 'raster',
                        source: 'opentopo',
                        minzoom: 0,
                        maxzoom: 22,
                    },
                ],
            },
            center: [-71.83579, -15.19149],
            zoom: 12,
        });

        // Crear popup pero sin mostrarlo inicialmente
        popupRef.current = new maplibregl.Popup({
            closeButton: false,
            closeOnClick: false,
        });

        mapRef.current.addControl(new maplibregl.NavigationControl(), 'top-right');
        mapRef.current.addControl(new maplibregl.ScaleControl(), 'bottom-right');

        mapRef.current.on('load', () => {
            mapRef.current.addSource('leyes', {
                type: 'geojson',
                data: {
                    type: 'FeatureCollection',
                    features: [],
                },
            });

            mapRef.current.addLayer({
                id: 'heatmap-layer',
                type: 'heatmap',
                source: 'leyes',
                maxzoom: 21,
                paint: {
                    'heatmap-weight': [
                        'interpolate',
                        ['linear'],
                        ['get', 'concentration'],
                        0,
                        0,
                        elementRanges['Cu'].max,
                        1,
                    ],
                    'heatmap-color': [
                        'interpolate',
                        ['linear'],
                        ['heatmap-density'],
                        0,
                        'rgba(0,0,255,0)',
                        0.2,
                        'blue',
                        0.4,
                        'green',
                        0.6,
                        'yellow',
                        0.8,
                        'orange',
                        1,
                        'red',
                    ],
                    'heatmap-radius': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        0,
                        10,
                        12,
                        20,
                        15,
                        30,
                        17,
                        40,
                    ],
                    'heatmap-intensity': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        0,
                        1,
                        12,
                        2,
                        15,
                        3,
                        17,
                        4,
                    ],
                    'heatmap-opacity': 0.9,
                },
            });

            mapRef.current.addLayer({
                id: 'circle-layer',
                type: 'circle',
                source: 'leyes',
                paint: {
                    'circle-radius': 2, // más pequeño, más parecido a un punto
                    'circle-color': '#1e1e1e', // color negro o algo muy visible
                    'circle-stroke-width': 0
                }
            });

            loadData(element);
            setMapLoaded(true);
        });

        return () => {
            if (mapRef.current) mapRef.current.remove();
        };
    }, []);

    // Debounce simple y actualización rango de heatmap-weight
    useEffect(() => {
        if (!mapRef.current || !mapRef.current.getSource('leyes')) return;

        const maxConcentration = elementRanges[element]?.max || 1000;
        mapRef.current.setPaintProperty('heatmap-layer', 'heatmap-weight', [
            'interpolate',
            ['linear'],
            ['get', 'concentration'],
            0,
            0,
            maxConcentration,
            1,
        ]);

        const handler = setTimeout(() => {
            loadData(element);
        }, 300);

        return () => clearTimeout(handler);
    }, [element]);

    useEffect(() => {
        const map = mapRef.current;

        if (!map || !popupRef.current || !mapLoaded || !map.getLayer('circle-layer')) return;

        const showPopup = (e) => {
            if (!e.features || e.features.length === 0) return;

            const feature = e.features[0];
            const coords = feature.geometry.coordinates;
            const concentration = feature.properties.concentration;

            popupRef.current
                .setLngLat(coords)
                .setHTML(`<strong>${element}</strong><br/>Concentración: ${concentration} ${elementRanges[element]?.unit || 'ppm'}`)
                .addTo(map);
        };

        const hidePopup = () => popupRef.current.remove();

        const onMouseDown = () => {
            map.getCanvas().style.cursor = 'grabbing'; // Al empezar a mover
        };

        const onMouseUp = () => {
            map.getCanvas().style.cursor = 'default'; // Al soltar
        };

        // Remover eventos previos
        map.off('mousemove', 'circle-layer', showPopup);
        map.off('mouseleave', 'circle-layer', hidePopup);
        map.off('mousedown', onMouseDown);
        map.off('mouseup', onMouseUp);

        // Añadir eventos
        map.on('mousemove', 'circle-layer', showPopup);
        map.on('mouseleave', 'circle-layer', hidePopup);
        map.on('mousedown', onMouseDown);
        map.on('mouseup', onMouseUp);

        // Cursor por defecto siempre al cargar
        map.getCanvas().style.cursor = 'default';

        return () => {
            map.off('mousemove', 'circle-layer', showPopup);
            map.off('mouseleave', 'circle-layer', hidePopup);
            map.off('mousedown', onMouseDown);
            map.off('mouseup', onMouseUp);
        };
    }, [element, mapLoaded]);



    function loadData(element) {
        if (cache.current[element]) {
            const source = mapRef.current.getSource('leyes');
            if (source) {
                source.setData(cache.current[element]);
            }
            setLoading(false);
            return;
        }

        setLoading(true);
        fetch(`http://127.0.0.1:8000/api/heatmap?element=${element}`)
            .then((res) => res.json())
            .then((geojson) => {
                cache.current[element] = geojson;
                const source = mapRef.current.getSource('leyes');
                if (source) {
                    source.setData(geojson);
                }
            })
            .catch(console.error)
            .finally(() => setLoading(false));
    }

    return (
        <div>
            <div
                style={{
                    position: 'absolute',
                    zIndex: 2,
                    padding: 10,
                    background: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    borderRadius: 5,
                    boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
                    fontSize: '14px',
                    userSelect: 'none',
                }}
            >
                <label>
                    Elemento:
                    <select
                        value={element}
                        onChange={(e) => setElement(e.target.value)}
                        style={{ marginLeft: 5 }}
                    >
                        <option value="Cu">Cu</option>
                        <option value="Au">Au</option>
                        <option value="Zn">Zn</option>
                        <option value="Pb">Pb</option>
                    </select>
                </label>
                {loading && <span>Cargando datos...</span>}
            </div>

            <div
                style={{
                    position: 'absolute',
                    zIndex: 2,
                    padding: 10,
                    bottom: 10,
                    background: 'rgba(255, 255, 255, 0.85)',
                    borderRadius: 5,
                    fontSize: '14px',
                    boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
                    userSelect: 'none',
                }}
            >
                <div>
                    <strong>Rango concentración ({element}):</strong>
                </div>
                <div>
                    Min: {elementRanges[element]?.min} {elementRanges[element]?.unit} - Max:{' '}
                    {elementRanges[element]?.max} {elementRanges[element]?.unit}
                </div>
            </div>

            <div ref={mapContainer} style={{ width: '100%', height: '100vh' }} />
        </div>
    );
}

export default HeatMapLibre;