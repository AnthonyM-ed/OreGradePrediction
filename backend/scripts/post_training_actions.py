"""
Post-Training Action Plan

This script outlines what to do after training the priority elements (CU, AU, AG).
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_models.inference.predictor import SpatialOreGradePredictor
from ml_models.inference.batch_predictor import BatchSpatialPredictor
from ml_models.training.train_pipeline import TrainingPipeline


class PostTrainingActions:
    """Actions to take after training priority elements"""
    
    def __init__(self):
        self.models_dir = Path("data/models")
        self.trained_elements = []
        self.model_performance = {}
        
    def analyze_trained_models(self):
        """Analyze performance of trained models"""
        print("📊 ANALYZING TRAINED MODELS")
        print("="*50)
        
        # Find all trained models
        model_files = list(self.models_dir.glob("model_metadata_*.json"))
        
        if not model_files:
            print("❌ No trained models found!")
            return
        
        print(f"Found {len(model_files)} trained models:\n")
        
        for metadata_file in sorted(model_files):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extract element from pipeline results
                element = "Unknown"
                if 'element' in metadata:
                    element = metadata['element']
                elif 'evaluation_results' in metadata:
                    # Try to extract from config or other sources
                    element = "Element"
                
                self.trained_elements.append(element)
                
                # Get performance metrics
                eval_results = metadata.get('evaluation_results', {})
                test_metrics = eval_results.get('test_metrics', {})
                
                r2_score = test_metrics.get('r2_score', 0)
                rmse = test_metrics.get('rmse', 0)
                mae = test_metrics.get('mae', 0)
                
                self.model_performance[element] = {
                    'r2_score': r2_score,
                    'rmse': rmse,
                    'mae': mae,
                    'timestamp': metadata.get('timestamp', 'Unknown'),
                    'data_records': metadata.get('data_records', 0),
                    'features_count': metadata.get('features_count', 0)
                }
                
                print(f"🧪 {element}:")
                print(f"   📈 R² Score: {r2_score:.4f}")
                print(f"   📊 RMSE: {rmse:.2f}")
                print(f"   📉 MAE: {mae:.2f}")
                print(f"   📁 Records: {metadata.get('data_records', 'N/A')}")
                print(f"   🔧 Features: {metadata.get('features_count', 'N/A')}")
                print(f"   📅 Trained: {metadata.get('timestamp', 'Unknown')}")
                print()
                
            except Exception as e:
                print(f"❌ Error reading {metadata_file}: {str(e)}")
        
        return self.model_performance
    
    def test_predictions(self):
        """Test predictions with trained models"""
        print("\n🔮 TESTING PREDICTIONS")
        print("="*50)
        
        if not self.trained_elements:
            print("❌ No trained elements found!")
            return
        
        # Test coordinates (example mining site)
        test_coordinates = [
            {"latitude": -23.5505, "longitude": -46.6333, "depth_from": 50, "depth_to": 55},
            {"latitude": -23.5515, "longitude": -46.6343, "depth_from": 100, "depth_to": 105},
            {"latitude": -23.5525, "longitude": -46.6353, "depth_from": 150, "depth_to": 155}
        ]
        
        print("Test coordinates:")
        for i, coord in enumerate(test_coordinates, 1):
            print(f"  {i}. Lat: {coord['latitude']}, Lon: {coord['longitude']}, Depth: {coord['depth_from']}-{coord['depth_to']}m")
        
        print("\nPredictions:")
        
        # Test each trained element
        for element in self.trained_elements:
            print(f"\n🧪 {element} Predictions:")
            
            try:
                # Initialize predictor
                predictor = SpatialOreGradePredictor()
                
                # Make predictions for each test coordinate
                for i, coord in enumerate(test_coordinates, 1):
                    try:
                        prediction = predictor.predict_at_location(
                            latitude=coord['latitude'],
                            longitude=coord['longitude'],
                            depth_from=coord['depth_from'],
                            depth_to=coord['depth_to'],
                            element=element
                        )
                        
                        print(f"   Point {i}: {prediction['predicted_grade']:.2f} ppm")
                        
                    except Exception as e:
                        print(f"   Point {i}: Error - {str(e)}")
                        
            except Exception as e:
                print(f"   ❌ Error initializing predictor for {element}: {str(e)}")
    
    def create_production_api_endpoints(self):
        """Show how to create production API endpoints"""
        print("\n🚀 PRODUCTION API ENDPOINTS")
        print("="*50)
        
        api_code = '''
# Add to your Django views.py or FastAPI routes

from ml_models.inference.real_time_predictor import RealTimePredictor
from ml_models.inference.batch_predictor import BatchSpatialPredictor

# Single prediction endpoint
@api_view(['POST'])
def predict_ore_grade(request):
    """
    Predict ore grade at specific coordinates
    
    POST data:
    {
        "latitude": -23.5505,
        "longitude": -46.6333,
        "depth_from": 50,
        "depth_to": 55,
        "element": "CU"
    }
    """
    try:
        predictor = RealTimePredictor()
        
        result = predictor.predict_grade(
            latitude=request.data['latitude'],
            longitude=request.data['longitude'],
            depth_from=request.data['depth_from'],
            depth_to=request.data['depth_to'],
            element=request.data['element']
        )
        
        return Response(result, status=200)
        
    except Exception as e:
        return Response({'error': str(e)}, status=400)

# Batch prediction endpoint
@api_view(['POST'])
def batch_predict_ore_grades(request):
    """
    Batch predict ore grades for multiple coordinates
    
    POST data:
    {
        "coordinates": [
            {"latitude": -23.5505, "longitude": -46.6333, "depth_from": 50, "depth_to": 55},
            {"latitude": -23.5515, "longitude": -46.6343, "depth_from": 100, "depth_to": 105}
        ],
        "elements": ["CU", "AU", "AG"]
    }
    """
    try:
        batch_predictor = BatchSpatialPredictor()
        
        results = batch_predictor.predict_multiple_locations(
            coordinates=request.data['coordinates'],
            elements=request.data['elements']
        )
        
        return Response(results, status=200)
        
    except Exception as e:
        return Response({'error': str(e)}, status=400)
'''
        
        print("📝 Example API endpoints code:")
        print(api_code)
        
        # Save to file
        api_file = Path("api_endpoints_example.py")
        with open(api_file, 'w') as f:
            f.write(api_code)
        
        print(f"\n📁 Code saved to: {api_file}")
    
    def suggest_next_training_batch(self):
        """Suggest next elements to train"""
        print("\n🎯 NEXT TRAINING RECOMMENDATIONS")
        print("="*50)
        
        # All available elements (based on mining industry importance)
        all_elements = [
            'CU', 'AU', 'AG',  # Already trained (priority)
            'PB', 'ZN',        # Next priority (base metals)
            'MO', 'FE',        # Industrial metals
            'S', 'AS', 'SB',   # Process indicators
            'NI', 'CO', 'W',   # Specialty metals
            'BI', 'CD', 'HG',  # Trace elements
            'SN', 'TI', 'V'    # Additional elements
        ]
        
        trained = set(self.trained_elements)
        remaining = [e for e in all_elements if e not in trained]
        
        print(f"✅ Already trained: {', '.join(trained)}")
        print(f"⏳ Next recommended batch: {', '.join(remaining[:3])}")
        print(f"🔮 Future batches: {', '.join(remaining[3:])}")
        
        # Calculate estimated time for next batch
        time_per_element = 247  # minutes (from estimation)
        next_batch_time = time_per_element * 3  # 3 elements
        
        print(f"\n⏱️ Estimated time for next batch (3 elements): {next_batch_time:.0f} minutes ({next_batch_time/60:.1f} hours)")
        
        return remaining[:3]
    
    def create_model_comparison_report(self):
        """Create comprehensive model comparison report"""
        print("\n📊 MODEL COMPARISON REPORT")
        print("="*50)
        
        if not self.model_performance:
            print("❌ No model performance data available!")
            return
        
        # Create comparison table
        print("Element | R² Score | RMSE    | MAE     | Records | Features")
        print("-" * 60)
        
        for element, metrics in self.model_performance.items():
            print(f"{element:7} | {metrics['r2_score']:8.4f} | {metrics['rmse']:7.2f} | {metrics['mae']:7.2f} | {metrics['data_records']:7} | {metrics['features_count']:8}")
        
        # Find best performing model
        best_element = max(self.model_performance.keys(), key=lambda e: self.model_performance[e]['r2_score'])
        best_r2 = self.model_performance[best_element]['r2_score']
        
        print(f"\n🏆 Best performing model: {best_element} (R² = {best_r2:.4f})")
        
        # Save detailed report
        report_file = Path("data/exports/reports/model_comparison_report.txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            for element, metrics in self.model_performance.items():
                f.write(f"\n{element}:\n")
                f.write(f"  R² Score: {metrics['r2_score']:.4f}\n")
                f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
                f.write(f"  MAE: {metrics['mae']:.2f}\n")
                f.write(f"  Records: {metrics['data_records']}\n")
                f.write(f"  Features: {metrics['features_count']}\n")
                f.write(f"  Trained: {metrics['timestamp']}\n")
        
        print(f"📄 Detailed report saved to: {report_file}")
    
    def run_complete_analysis(self):
        """Run complete post-training analysis"""
        print("🔍 POST-TRAINING ANALYSIS")
        print("="*60)
        
        # Step 1: Analyze trained models
        self.analyze_trained_models()
        
        # Step 2: Test predictions
        self.test_predictions()
        
        # Step 3: Create model comparison report
        self.create_model_comparison_report()
        
        # Step 4: Suggest next training batch
        next_batch = self.suggest_next_training_batch()
        
        # Step 5: Show production API endpoints
        self.create_production_api_endpoints()
        
        print("\n✅ POST-TRAINING ANALYSIS COMPLETE!")
        print("="*60)
        
        return {
            'trained_elements': self.trained_elements,
            'model_performance': self.model_performance,
            'next_batch': next_batch
        }


def main():
    """Main execution function"""
    try:
        analyzer = PostTrainingActions()
        results = analyzer.run_complete_analysis()
        
        print(f"\n🎉 SUMMARY:")
        print(f"✅ Analyzed {len(results['trained_elements'])} trained models")
        print(f"🎯 Next recommended batch: {', '.join(results['next_batch'])}")
        print(f"🚀 Ready for production deployment!")
        
    except Exception as e:
        print(f"❌ Error in post-training analysis: {str(e)}")


if __name__ == "__main__":
    main()
