import React, { useState } from 'react';
import './OreGradePrediction.css';

const OreGradePrediction = () => {
  const [predictionForm, setPredictionForm] = useState({
    element: 'CU',
    latitude: '',
    longitude: '',
    depth_from: '',
    depth_to: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Available elements
  const elements = [
    { code: 'CU', name: 'Copper (Cu)', color: '#B87333' },
    { code: 'AU', name: 'Gold (Au)', color: '#FFD700' },
    { code: 'AG', name: 'Silver (Ag)', color: '#C0C0C0' },
    { code: 'PB', name: 'Lead (Pb)', color: '#2F4F4F' },
    { code: 'ZN', name: 'Zinc (Zn)', color: '#708090' },
    { code: 'MO', name: 'Molybdenum (Mo)', color: '#778899' },
    { code: 'FE', name: 'Iron (Fe)', color: '#B22222' },
    { code: 'S', name: 'Sulfur (S)', color: '#FFFF00' },
    { code: 'AS', name: 'Arsenic (As)', color: '#696969' },
    { code: 'SB', name: 'Antimony (Sb)', color: '#A9A9A9' }
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPredictionForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict-ore-grade/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionForm)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(`Error making prediction: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const selectedElement = elements.find(el => el.code === predictionForm.element);

  return (
    <div className="ore-grade-prediction">
      <div className="prediction-header">
        <h1>Ore Grade Prediction</h1>
        <p>Predict ore grade concentrations using trained XGBoost models</p>
      </div>

      <div className="prediction-content">
        <div className="prediction-form-container">
          <form onSubmit={handlePredict} className="prediction-form">
            <h3>Prediction Parameters</h3>
            
            {/* Element Selection */}
            <div className="form-group">
              <label htmlFor="element">
                <span className="element-indicator" style={{ backgroundColor: selectedElement?.color }}></span>
                Element
              </label>
              <select
                id="element"
                name="element"
                value={predictionForm.element}
                onChange={handleInputChange}
                required
              >
                {elements.map(element => (
                  <option key={element.code} value={element.code}>
                    {element.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Coordinates */}
            <div className="coordinates-group">
              <div className="form-group half-width">
                <label htmlFor="latitude">Latitude</label>
                <input
                  type="number"
                  id="latitude"
                  name="latitude"
                  value={predictionForm.latitude}
                  onChange={handleInputChange}
                  step="0.000001"
                  placeholder="-19.123456"
                  required
                />
              </div>
              <div className="form-group half-width">
                <label htmlFor="longitude">Longitude</label>
                <input
                  type="number"
                  id="longitude"
                  name="longitude"
                  value={predictionForm.longitude}
                  onChange={handleInputChange}
                  step="0.000001"
                  placeholder="-69.654321"
                  required
                />
              </div>
            </div>

            {/* Depth Range */}
            <div className="depth-group">
              <div className="form-group half-width">
                <label htmlFor="depth_from">Depth From (m)</label>
                <input
                  type="number"
                  id="depth_from"
                  name="depth_from"
                  value={predictionForm.depth_from}
                  onChange={handleInputChange}
                  step="0.1"
                  min="0"
                  placeholder="0.0"
                  required
                />
              </div>
              <div className="form-group half-width">
                <label htmlFor="depth_to">Depth To (m)</label>
                <input
                  type="number"
                  id="depth_to"
                  name="depth_to"
                  value={predictionForm.depth_to}
                  onChange={handleInputChange}
                  step="0.1"
                  min="0"
                  placeholder="10.0"
                  required
                />
              </div>
            </div>

            <button 
              type="submit" 
              className="predict-button"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Predicting...
                </>
              ) : (
                <>
                  Predict Ore Grade
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Section */}
        <div className="prediction-results-container">
          {error && (
            <div className="prediction-error">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          )}

          {prediction && (
            <div className="prediction-results">
              <h3>Prediction Results</h3>
              
              <div className="result-main">
                <div className="predicted-grade">
                  <span className="grade-label">Predicted Grade:</span>
                  <span 
                    className="grade-value"
                    style={{ color: selectedElement?.color }}
                  >
                    {prediction.predicted_grade?.toFixed(2)} ppm
                  </span>
                </div>
              </div>

              <div className="result-details">
                <div className="detail-item">
                  <span className="detail-label">Element:</span>
                  <span className="detail-value">{prediction.element}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Location:</span>
                  <span className="detail-value">
                    {prediction.latitude?.toFixed(6)}, {prediction.longitude?.toFixed(6)}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Depth Range:</span>
                  <span className="detail-value">
                    {prediction.depth_from} - {prediction.depth_to} m
                  </span>
                </div>
                {prediction.confidence && (
                  <div className="detail-item">
                    <span className="detail-label">Confidence:</span>
                    <span className="detail-value">{prediction.confidence?.toFixed(3)}</span>
                  </div>
                )}
                {prediction.model_info && (
                  <div className="detail-item">
                    <span className="detail-label">Model:</span>
                    <span className="detail-value">{prediction.model_info}</span>
                  </div>
                )}
              </div>

              {/* Grade Classification 
              {prediction.predicted_grade && (
                <div className="grade-classification">
                  <h4>Grade Classification:</h4>
                  <div className="classification-bar">
                    <div 
                      className={`classification-segment ${getGradeClassification(prediction.predicted_grade, predictionForm.element).class}`}
                    >
                      {getGradeClassification(prediction.predicted_grade, predictionForm.element).label}
                    </div>
                  </div>
                  <p className="classification-description">
                    {getGradeClassification(prediction.predicted_grade, predictionForm.element).description}
                  </p>
                </div>
              )}*/}
            </div>
          )}

          {/* Usage Tips */}
          <div className="usage-tips">
            <h4>Usage Tips:</h4>
            <ul>
              <li>Enter coordinates in decimal degrees format</li>
              <li>Depth should be specified from surface downward (positive values)</li>
              <li>Models are trained on real geological data</li>
              <li>Results are estimates based on spatial patterns</li>
              <li>Higher accuracy in areas with more training data</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function to classify grade levels
const getGradeClassification = (grade, element) => {
  const thresholds = {
    'CU': { high: 5000, medium: 1000, low: 100 },
    'AU': { high: 10, medium: 2, low: 0.5 },
    'AG': { high: 100, medium: 20, low: 5 },
    'PB': { high: 10000, medium: 2000, low: 500 },
    'ZN': { high: 20000, medium: 5000, low: 1000 },
    'MO': { high: 1000, medium: 200, low: 50 },
    'FE': { high: 300000, medium: 100000, low: 20000 },
    'S': { high: 50000, medium: 10000, low: 2000 },
    'AS': { high: 1000, medium: 200, low: 50 },
    'SB': { high: 500, medium: 100, low: 20 }
  };

  const threshold = thresholds[element] || thresholds['CU'];

  if (grade >= threshold.high) {
    return {
      class: 'high-grade',
      label: 'High Grade',
      description: 'Excellent economic potential - highly valuable deposit'
    };
  } else if (grade >= threshold.medium) {
    return {
      class: 'medium-grade',
      label: 'Medium Grade',
      description: 'Good economic potential - viable for extraction'
    };
  } else if (grade >= threshold.low) {
    return {
      class: 'low-grade',
      label: 'Low Grade',
      description: 'Marginal economic potential - may require bulk mining'
    };
  } else {
    return {
      class: 'trace-grade',
      label: 'Trace Grade',
      description: 'Below economic threshold - not suitable for mining'
    };
  }
};

export default OreGradePrediction;
