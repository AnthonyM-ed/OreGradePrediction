import React, { useState, useRef, useEffect } from 'react';
import './InteractiveOreGradePrediction.css';

const InteractiveOreGradePrediction = () => {
  const mapRef = useRef(null);
  const [map, setMap] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [predictionForm, setPredictionForm] = useState({
    element: 'CU',
    depth_from: '',
    depth_to: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // Initialize map (using Leaflet as example)
  useEffect(() => {
    // Map initialization code would go here
    // For now, this is a placeholder for the enhanced version
    console.log('Map initialization placeholder');
  }, []);

  const handleMapClick = (event) => {
    // Handle map click to set coordinates
    const { lat, lng } = event.latlng;
    setSelectedPoint({ latitude: lat, longitude: lng });
    console.log('Map clicked at:', lat, lng);
  };

  const handlePredict = async () => {
    if (!selectedPoint) {
      alert('Please select a point on the map first');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict-ore-grade', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...predictionForm,
          latitude: selectedPoint.latitude,
          longitude: selectedPoint.longitude
        })
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="interactive-prediction">
      <div className="map-container">
        <h2>🗺️ Interactive Map - Select Location</h2>
        <div ref={mapRef} className="map-placeholder">
          <p>Interactive map will be implemented here</p>
          <p>Click on the map to select coordinates for prediction</p>
          {selectedPoint && (
            <div className="selected-coordinates">
              <strong>Selected Point:</strong><br />
              Lat: {selectedPoint.latitude.toFixed(6)}<br />
              Lng: {selectedPoint.longitude.toFixed(6)}
            </div>
          )}
        </div>
      </div>

      <div className="prediction-panel">
        <h3>Prediction Parameters</h3>
        
        <div className="form-group">
          <label>Element:</label>
          <select
            value={predictionForm.element}
            onChange={(e) => setPredictionForm(prev => ({
              ...prev,
              element: e.target.value
            }))}
          >
            <option value="CU">Copper (Cu)</option>
            <option value="AU">Gold (Au)</option>
            <option value="AG">Silver (Ag)</option>
            <option value="PB">Lead (Pb)</option>
            <option value="ZN">Zinc (Zn)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Depth From (m):</label>
          <input
            type="number"
            value={predictionForm.depth_from}
            onChange={(e) => setPredictionForm(prev => ({
              ...prev,
              depth_from: e.target.value
            }))}
            placeholder="0.0"
          />
        </div>

        <div className="form-group">
          <label>Depth To (m):</label>
          <input
            type="number"
            value={predictionForm.depth_to}
            onChange={(e) => setPredictionForm(prev => ({
              ...prev,
              depth_to: e.target.value
            }))}
            placeholder="10.0"
          />
        </div>

        <button 
          onClick={handlePredict}
          disabled={!selectedPoint || loading}
          className="predict-button"
        >
          {loading ? 'Predicting...' : 'Predict Ore Grade'}
        </button>

        {prediction && (
          <div className="prediction-result">
            <h4>Prediction Result:</h4>
            <p><strong>Grade:</strong> {prediction.predicted_grade?.toFixed(2)} ppm</p>
            <p><strong>Element:</strong> {prediction.element}</p>
            <p><strong>Location:</strong> {prediction.latitude?.toFixed(6)}, {prediction.longitude?.toFixed(6)}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default InteractiveOreGradePrediction;
