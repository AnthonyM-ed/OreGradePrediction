"""
Training Progress Monitor

This script monitors the training progress and provides real-time updates.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training progress and system resources"""
    
    def __init__(self):
        self.models_dir = Path("data/models")
        self.reports_dir = Path("data/exports/reports")
        self.start_time = None
        self.last_check = None
        
    def get_current_models(self):
        """Get list of currently trained models"""
        if not self.models_dir.exists():
            return []
        
        # Find all model files
        model_files = list(self.models_dir.glob("grade_model_*.joblib"))
        
        models = []
        for model_file in model_files:
            # Extract timestamp from filename
            timestamp = model_file.stem.split('_')[-2] + '_' + model_file.stem.split('_')[-1]
            
            # Get corresponding metadata
            metadata_file = self.models_dir / f"model_metadata_{timestamp}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    models.append({
                        'timestamp': timestamp,
                        'model_file': str(model_file),
                        'metadata': metadata
                    })
                except:
                    pass
        
        return sorted(models, key=lambda x: x['timestamp'])
    
    def get_training_statistics(self):
        """Get training statistics"""
        models = self.get_current_models()
        
        if not models:
            return {
                'total_models': 0,
                'elements_trained': [],
                'latest_model': None,
                'training_active': False
            }
        
        # Extract elements from models
        elements = []
        for model in models:
            element = model['metadata'].get('element', 'Unknown')
            elements.append(element)
        
        # Check if training is active (model created in last 5 minutes)
        latest_model = models[-1]
        latest_timestamp = datetime.fromisoformat(latest_model['metadata'].get('timestamp', ''))
        now = datetime.now()
        time_diff = (now - latest_timestamp).total_seconds()
        training_active = time_diff < 300  # 5 minutes
        
        return {
            'total_models': len(models),
            'elements_trained': elements,
            'latest_model': latest_model,
            'training_active': training_active,
            'time_since_last': time_diff
        }
    
    def format_time(self, seconds):
        """Format seconds into readable time"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def display_progress(self):
        """Display current training progress"""
        stats = self.get_training_statistics()
        
        print("\n" + "="*60)
        print("🔍 TRAINING PROGRESS MONITOR")
        print("="*60)
        
        print(f"📊 Total Models: {stats['total_models']}")
        print(f"🧪 Elements Trained: {', '.join(stats['elements_trained'])}")
        
        if stats['latest_model']:
            latest = stats['latest_model']
            metadata = latest['metadata']
            
            print(f"📅 Latest Model: {metadata.get('element', 'Unknown')} ({latest['timestamp']})")
            print(f"⏱️  Time Since Last: {self.format_time(stats['time_since_last'])}")
            
            # Show performance if available
            eval_results = metadata.get('evaluation_results', {})
            test_metrics = eval_results.get('test_metrics', {})
            
            if test_metrics:
                print(f"📈 Latest Performance:")
                print(f"   R²: {test_metrics.get('r2_score', 'N/A'):.4f}")
                print(f"   RMSE: {test_metrics.get('rmse', 'N/A'):.2f}")
                print(f"   MAE: {test_metrics.get('mae', 'N/A'):.2f}")
        
        print(f"🔄 Training Active: {'Yes' if stats['training_active'] else 'No'}")
        
        # Show disk usage
        total_size = sum(f.stat().st_size for f in self.models_dir.glob("*") if f.is_file())
        print(f"💾 Models Disk Usage: {total_size / (1024*1024):.1f} MB")
        
        return stats
    
    def monitor_training(self, check_interval=30):
        """Monitor training progress continuously"""
        print("🚀 Starting Training Monitor...")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                stats = self.display_progress()
                
                # Check if training is complete (no new models for 10 minutes)
                if not stats['training_active'] and stats['time_since_last'] > 600:
                    print("\n✅ Training appears to be complete!")
                    break
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def generate_progress_report(self):
        """Generate a progress report"""
        stats = self.get_training_statistics()
        models = self.get_current_models()
        
        report = f"""
TRAINING PROGRESS REPORT
========================
Generated: {datetime.now().isoformat()}

OVERVIEW:
- Total Models: {stats['total_models']}
- Elements Trained: {', '.join(stats['elements_trained'])}
- Training Active: {'Yes' if stats['training_active'] else 'No'}

DETAILED MODEL LIST:
"""
        
        for model in models:
            metadata = model['metadata']
            eval_results = metadata.get('evaluation_results', {})
            test_metrics = eval_results.get('test_metrics', {})
            
            report += f"""
Element: {metadata.get('element', 'Unknown')}
Timestamp: {model['timestamp']}
Records: {metadata.get('data_records', 'N/A')}
Features: {metadata.get('features_count', 'N/A')}
"""
            
            if test_metrics:
                report += f"R²: {test_metrics.get('r2_score', 'N/A'):.4f}\n"
                report += f"RMSE: {test_metrics.get('rmse', 'N/A'):.2f}\n"
                report += f"MAE: {test_metrics.get('mae', 'N/A'):.2f}\n"
        
        return report


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--continuous', '-c', action='store_true', help='Monitor continuously')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--report', '-r', action='store_true', help='Generate progress report')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor()
    
    if args.continuous:
        monitor.monitor_training(args.interval)
    elif args.report:
        report = monitor.generate_progress_report()
        print(report)
        
        # Save report
        reports_dir = Path("data/exports/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"progress_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: {report_file}")
    else:
        monitor.display_progress()


if __name__ == "__main__":
    main()
