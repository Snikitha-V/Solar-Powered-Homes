# ============================================
# STEP 2: TRAIN YOLOv11 - OPTIMIZED FOR A100 12GB
# train_yolo_optimized.py
# ============================================

from ultralytics import YOLO
import torch
import mlflow
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import yaml
import shutil
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

def log(msg):
    """Print with flush to ensure terminal output."""
    print(msg, flush=True)
    sys.stdout.flush()

# Torch 2.1 lacks torch.utils._pytree.register_pytree_node but newer libs expect it.
try:
    from torch.utils import _pytree as torch_pytree

    if not hasattr(torch_pytree, 'register_pytree_node') and hasattr(torch_pytree, '_register_pytree_node'):
        def _compat_register(node_type, flatten_fn, unflatten_fn, *, serialized_type_name=None, serialized_context_fn=None):
            # Older torch ignores serialization kwargs, so drop them for compatibility.
            return torch_pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

        torch_pytree.register_pytree_node = _compat_register
except Exception:
    pass

def train_yolo_a100_optimized():
    """
    Train YOLOv11 optimized for A100 12GB VRAM + 16GB RAM
    
    Key optimizations:
    - Batch size tuned for 12GB VRAM
    - Mixed precision training (AMP)
    - Gradient accumulation if needed
    - Memory-efficient augmentations
    """
    
    print("=" * 80)
    print("TRAINING YOLOv11 - OPTIMIZED FOR A100 12GB")
    print("=" * 80)
    
    # Verify CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Please check GPU setup.")
    
    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Dataset config
    data_yaml = Path("data/merged_solar_dataset/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}\nRun merge_datasets.py first!")
    
    # Training configuration optimized for A100 12GB
    config = {
        # Model selection (choose one):
        # 'yolov11n.pt' - Fastest, lowest VRAM (2-3GB), F1~0.85
        # 'yolov11s.pt' - Balanced, medium VRAM (4-5GB), F1~0.87
        # 'yolov11m.pt' - Better accuracy, higher VRAM (7-8GB), F1~0.89
        # 'yolov11l.pt' - Best accuracy, max VRAM (10-11GB), F1~0.91
        
        'model': str(Path('trained_model/yolov11_solar.pt')),  # Finetune from current solar weights
        
        # Training params
            'epochs': 60,
            'imgsz': 1536,
            'batch': 1,  # Smaller batch to fit larger images
        'workers': 0,  # Workaround: single-process dataloader to avoid long startup
        'patience': 20,  # Early stopping
        'save_period': 5,
        
        # Performance
        'amp': True,  # Mixed precision (faster + less VRAM)
        'cache': False,  # Don't cache in RAM (only 16GB available)
        'device': 0,  # GPU 0
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.0003,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0003,
        'warmup_epochs': 1.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation (optimized for rooftop solar)
        'hsv_h': 0.01,   # Softer color jitter
        'hsv_s': 0.5,
        'hsv_v': 0.35,
        'degrees': 5.0,
        'translate': 0.08,
        'scale': 0.3,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.2,
        'fliplr': 0.5,
        'mosaic': 0.2,   # Limited mosaic to avoid artifacts on small roofs
        'mixup': 0.05,
        'copy_paste': 0.0,
        
        # Loss weights (fine-tuned for detection)
        'box': 7.5,
        'cls': 0.7,
        'dfl': 1.5,
    }
    
    model_name = Path(config['model']).name

    # Adjust batch size based on model selection
    if model_name == 'yolov11n.pt':
        config['batch'] = 64
    elif model_name == 'yolov11s.pt':
        config['batch'] = 48
    elif model_name == 'yolov11m.pt':
        config['batch'] = 24
    elif model_name == 'yolov11l.pt':
        config['batch'] = 16
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Batch Size: {config['batch']}")
    print(f"  Image Size: {config['imgsz']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Mixed Precision: {config['amp']}")
    
    # Initialize MLflow
    mlflow.set_experiment("solar-detection-training")
    mlflow.set_tracking_uri("file:./mlruns")
    
    with mlflow.start_run(run_name=f"yolo_{config['model'].replace('.pt', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log all hyperparameters
        mlflow.log_params(config)
        
        # Load model
        print(f"\n{'='*60}")
        print("LOADING MODEL")
        print('='*60)
        model = YOLO(config['model'])
        
        # Start training
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print('='*60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Expected duration: ~{config['epochs'] * 0.5:.0f} minutes for {config['epochs']} epochs")
        
        results = model.train(
            # Data
            data=str(data_yaml),
            
            # Training
            epochs=config['epochs'],
            patience=config['patience'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            
            # Performance
            device=config['device'],
            workers=config['workers'],
            amp=config['amp'],
            cache=config['cache'],
            
            # Saving
            save=True,
            save_period=config['save_period'],
            project='runs/detect',
            name='solar_yolo',
            exist_ok=True,
            
            # Optimizer
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            
            # Augmentation
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            degrees=config['degrees'],
            translate=config['translate'],
            scale=config['scale'],
            shear=config['shear'],
            perspective=config['perspective'],
            flipud=config['flipud'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            mixup=config['mixup'],
            copy_paste=config['copy_paste'],
            
            # Loss weights
            box=config['box'],
            cls=config['cls'],
            dfl=config['dfl'],
            
            # Misc
            pretrained=True,
            verbose=True,
            seed=42,
            deterministic=True,
            plots=True,
            val=True,
            profile=False,  # Disable profiling to avoid extra overhead
        )
        
        # Print real-time epoch progress from training results
        if hasattr(results, 'results_dict'):
            print(f"\n{'='*60}")
            print("EPOCH RESULTS SUMMARY")
            print('='*60)
            for key, val in results.results_dict.items():
                if isinstance(val, (int, float)):
                    print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print('='*60)
        
        # Log final metrics to MLflow
        metrics = {
            'final_mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'final_mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'final_precision': float(results.results_dict.get('metrics/precision(B)', 0)),
            'final_recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        }
        
        # Calculate F1 score
        if metrics['final_precision'] > 0 and metrics['final_recall'] > 0:
            f1_score = 2 * (metrics['final_precision'] * metrics['final_recall']) / \
                      (metrics['final_precision'] + metrics['final_recall'])
            metrics['final_f1_score'] = f1_score
        
        mlflow.log_metrics(metrics)
        
        # Validate on test set
        print(f"\n{'='*60}")
        print("VALIDATING ON TEST SET")
        print('='*60)
        
        test_results = model.val(
            data=str(data_yaml),
            split='test',
            batch=config['batch'],
            imgsz=config['imgsz'],
            save_json=True,
            plots=True,
        )
        
        test_metrics = {
            'test_mAP50': float(test_results.results_dict.get('metrics/mAP50(B)', 0)),
            'test_mAP50-95': float(test_results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'test_precision': float(test_results.results_dict.get('metrics/precision(B)', 0)),
            'test_recall': float(test_results.results_dict.get('metrics/recall(B)', 0)),
        }
        
        # Calculate test F1
        if test_metrics['test_precision'] > 0 and test_metrics['test_recall'] > 0:
            test_f1 = 2 * (test_metrics['test_precision'] * test_metrics['test_recall']) / \
                     (test_metrics['test_precision'] + test_metrics['test_recall'])
            test_metrics['test_f1_score'] = test_f1
        
        mlflow.log_metrics(test_metrics)
        
        # Save best model to trained_model/
        best_model_path = Path("runs/detect/solar_yolo/weights/best.pt")
        output_path = Path("trained_model/yolov11_solar.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if best_model_path.exists():
            shutil.copy2(best_model_path, output_path)
            print(f"\n✓ Best model saved to: {output_path}")
        
        # Save training config
        config_path = Path("trained_model/training_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Export training logs to CSV (REQUIRED FOR SUBMISSION)
        export_training_logs_to_csv()
        
        # Print final results
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print('='*80)
        print(f"\nValidation Metrics:")
        print(f"  mAP50:     {metrics['final_mAP50']:.4f}")
        print(f"  mAP50-95:  {metrics['final_mAP50-95']:.4f}")
        print(f"  Precision: {metrics['final_precision']:.4f}")
        print(f"  Recall:    {metrics['final_recall']:.4f}")
        if 'final_f1_score' in metrics:
            print(f"  F1 Score:  {metrics['final_f1_score']:.4f} ⭐")
        
        print(f"\nTest Metrics:")
        print(f"  mAP50:     {test_metrics['test_mAP50']:.4f}")
        print(f"  mAP50-95:  {test_metrics['test_mAP50-95']:.4f}")
        print(f"  Precision: {test_metrics['test_precision']:.4f}")
        print(f"  Recall:    {test_metrics['test_recall']:.4f}")
        if 'test_f1_score' in test_metrics:
            print(f"  F1 Score:  {test_metrics['test_f1_score']:.4f} ⭐")
        
        print(f"\nModel saved to: {output_path}")
        print(f"Training logs exported to: training_logs/")
        
        return output_path, test_metrics

def export_training_logs_to_csv():
    """
    Export training logs to CSV format (REQUIRED FOR SUBMISSION)
    """
    
    print(f"\n{'='*60}")
    print("EXPORTING TRAINING LOGS")
    print('='*60)
    
    # Read results from YOLOv11 training
    results_csv = Path("runs/detect/solar_yolo/results.csv")
    
    if results_csv.exists():
        # Load YOLO results
        df = pd.read_csv(results_csv)
        df = df.rename(columns=lambda x: x.strip())  # Remove whitespace
        
        # Create simplified metrics CSV for submission
        metrics_df = pd.DataFrame({
            'epoch': df['epoch'],
            'train_loss': df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss'],
            'val_loss': df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss'],
            'precision': df['metrics/precision(B)'],
            'recall': df['metrics/recall(B)'],
            'mAP50': df['metrics/mAP50(B)'],
            'mAP50-95': df['metrics/mAP50-95(B)'],
        })
        
        # Calculate F1 score
        metrics_df['f1_score'] = 2 * (metrics_df['precision'] * metrics_df['recall']) / \
                                 (metrics_df['precision'] + metrics_df['recall'])
        
        # Add RMSE placeholder (will be calculated during inference)
        metrics_df['rmse_area'] = 0.0
        
        # Save to training_logs/
        output_dir = Path("training_logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "training_metrics.csv"
        metrics_df.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"✓ Training metrics saved to: {output_path}")
        
        # Also save as JSON for MLflow export
        mlflow_dir = output_dir / "mlflow_export"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        
        final_metrics = {
            'epoch': int(metrics_df.iloc[-1]['epoch']),
            'precision': float(metrics_df.iloc[-1]['precision']),
            'recall': float(metrics_df.iloc[-1]['recall']),
            'f1_score': float(metrics_df.iloc[-1]['f1_score']),
            'mAP50': float(metrics_df.iloc[-1]['mAP50']),
            'mAP50-95': float(metrics_df.iloc[-1]['mAP50-95']),
        }
        
        with open(mlflow_dir / "metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"✓ MLflow export saved to: {mlflow_dir}")
        
        return output_path
    else:
        print(f"⚠️  Results file not found: {results_csv}")
        return None

if __name__ == "__main__":
    train_yolo_a100_optimized()
