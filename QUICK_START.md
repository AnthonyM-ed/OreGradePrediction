# 🚀 Ore Grade Prediction System - Quick Start Guide

This guide will help you start the complete system with both backend and frontend.

## 🏁 Quick Start Steps

### 1. Start the Backend (Django)
```bash
cd backend
python manage.py runserver
```

### 2. Start the Frontend (React)
```bash
cd client
npm run dev
```

### 3. Test the System
```bash
cd backend
python test_prediction_api.py
```

## 🖥️ System URLs

- **Frontend Application**: http://localhost:5173
- **Backend API**: http://127.0.0.1:8000
- **Prediction Endpoint**: http://127.0.0.1:8000/api/predict-ore-grade/
- **Available Models**: http://127.0.0.1:8000/api/available-models/

## 🎯 How to Use the Prediction System

1. **Open the Frontend**: Navigate to http://localhost:5173
2. **Go to AI Prediction**: Click "AI Prediction" in the sidebar
3. **Fill the Form**:
   - Select element (CU, AU, AG, etc.)
   - Enter latitude/longitude coordinates
   - Enter depth range (from/to in meters)
   - Click "Predict Grade"

## 📊 Expected Results

Your trained models should predict ore grades with:
- **Accuracy**: ~91.33% R² score
- **Elements Available**: CU, AU, AG (based on your training)
- **Grade Classification**: 
  - Low: Below economic threshold
  - Medium: Near economic threshold  
  - High: Above economic threshold

## 🛠️ Troubleshooting

### Backend Issues
- **Port 8000 in use**: Change port with `python manage.py runserver 8001`
- **Missing models**: Run training scripts first
- **API errors**: Check Django logs in terminal

### Frontend Issues
- **Port 5173 in use**: Vite will auto-select another port
- **API connection**: Ensure backend is running first
- **No predictions**: Check test_prediction_api.py results

### Model Issues
- **No trained models**: Run `python quick_train_example.py`
- **Poor predictions**: Check model metadata files
- **Element not found**: Verify element is in trained models

## 📁 Key Files

### Frontend
- `client/src/components/OreGradePrediction.jsx` - Main prediction interface
- `client/src/components/OreGradePrediction.css` - Styling
- `client/src/App.jsx` - Main app with routing

### Backend
- `backend/api/views.py` - Prediction API endpoints
- `backend/api/urls.py` - API routing
- `backend/ml_models/inference/` - ML prediction system
- `backend/data/models/` - Trained model files

### Testing
- `backend/test_prediction_api.py` - API testing script
- `backend/scripts/test_complete_integration.py` - Full system test

## 🔧 Configuration

### Element Configuration
Edit `client/src/components/OreGradePrediction.jsx` to modify:
- Available elements
- Economic thresholds
- Grade classifications

### API Configuration
Edit `backend/api/views.py` to modify:
- Prediction logic
- Error handling
- Response format

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Grade Classification**: Visual color coding for grades
- **Loading States**: Progress indicators during prediction
- **Error Handling**: User-friendly error messages
- **Professional Styling**: Modern gradient design

## 🚀 Next Steps

1. **Test Current System**: Use the quick start guide above
2. **Train More Models**: Add more elements if needed
3. **Enhance UI**: Add interactive map functionality
4. **Optimize Models**: Improve accuracy for Au/Ag elements
5. **Add Features**: Export predictions, batch processing, etc.

## 💡 Tips

- **Testing**: Always run `test_prediction_api.py` before frontend testing
- **Development**: Keep both terminals open (backend + frontend)
- **Debugging**: Check browser console and Django terminal for errors
- **Models**: Keep model files in `backend/data/models/`
- **Updates**: Restart servers after code changes
