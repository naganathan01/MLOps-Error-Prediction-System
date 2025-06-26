"""
Additional API endpoints for the MLOps Error Prediction system.
This module extends the main app with additional functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    roc_auc: float = Field(..., ge=0, le=1)
    last_updated: str

class ModelRetraining(BaseModel):
    """Model retraining request"""
    model_types: Optional[List[str]] = None
    use_latest_data: bool = True
    notify_on_completion: bool = False

class DataDriftReport(BaseModel):
    """Data drift detection report"""
    drift_detected: bool
    drift_score: float = Field(..., ge=0, le=1)
    affected_features: List[str]
    recommendation: str
    timestamp: str

class SystemAlert(BaseModel):
    """System alert model"""
    alert_id: str
    severity: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    message: str
    timestamp: str
    resolved: bool = False

# Performance monitoring endpoints
@router.get("/monitoring/performance", response_model=List[ModelPerformance])
async def get_model_performance():
    """Get performance metrics for all models"""
    try:
        models_dir = Path("models")
        metadata_file = models_dir / "training_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="No training metadata found")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        performance_data = []
        
        if 'results' in metadata:
            for model_name, metrics in metadata['results'].items():
                performance = ModelPerformance(
                    model_name=model_name,
                    accuracy=metrics.get('accuracy', 0.0),
                    precision=metrics.get('precision', 0.0),
                    recall=metrics.get('recall', 0.0),
                    f1_score=metrics.get('f1_score', 0.0),
                    roc_auc=metrics.get('auc_score', 0.0),
                    last_updated=metadata.get('training_date', datetime.now().isoformat())
                )
                performance_data.append(performance)
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")

@router.get("/monitoring/drift", response_model=DataDriftReport)
async def detect_data_drift():
    """Detect data drift in recent predictions"""
    try:
        # This is a simplified drift detection
        # In production, you would compare recent data distributions with training data
        
        # Mock drift detection for demonstration
        # In reality, you would:
        # 1. Load recent prediction data
        # 2. Compare with training data distributions
        # 3. Calculate drift metrics (KS test, Population Stability Index, etc.)
        
        drift_score = np.random.uniform(0, 0.3)  # Mock score
        drift_detected = drift_score > 0.2
        
        affected_features = []
        if drift_detected:
            # Mock affected features
            all_features = ['cpu_usage', 'memory_usage', 'error_count', 'response_time_ms']
            affected_features = np.random.choice(all_features, size=2, replace=False).tolist()
        
        recommendation = "No action needed" if not drift_detected else "Consider model retraining"
        
        report = DataDriftReport(
            drift_detected=drift_detected,
            drift_score=round(drift_score, 3),
            affected_features=affected_features,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

# Model management endpoints
@router.post("/models/retrain")
async def trigger_model_retraining(
    retraining_request: ModelRetraining,
    background_tasks: BackgroundTasks
):
    """Trigger model retraining in the background"""
    try:
        # Add retraining task to background
        background_tasks.add_task(
            retrain_models_task,
            retraining_request.model_types,
            retraining_request.use_latest_data
        )
        
        return {
            "message": "Model retraining initiated",
            "models": retraining_request.model_types or ["all"],
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """Get current status of all models"""
    try:
        models_dir = Path("models")
        
        if not models_dir.exists():
            return {"status": "no_models", "models": []}
        
        model_files = list(models_dir.glob("*_model.joblib"))
        
        models_status = []
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            file_stats = model_file.stat()
            
            status = {
                "name": model_name,
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "status": "ready"
            }
            models_status.append(status)
        
        return {
            "status": "healthy" if models_status else "no_models",
            "total_models": len(models_status),
            "models": models_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

# Data management endpoints
@router.get("/data/summary")
async def get_data_summary():
    """Get summary of training and processed data"""
    try:
        data_dir = Path("data")
        summary = {
            "raw_data": {},
            "processed_data": {},
            "last_updated": None
        }
        
        # Check raw data
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            raw_files = {}
            for file_path in raw_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    raw_files[file_path.name] = {
                        "records": len(df),
                        "columns": len(df.columns),
                        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
                    }
                except Exception:
                    raw_files[file_path.name] = {"error": "Could not read file"}
            
            summary["raw_data"] = raw_files
        
        # Check processed data
        processed_dir = data_dir / "processed"
        if processed_dir.exists():
            processed_files = {}
            for file_path in processed_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    processed_files[file_path.name] = {
                        "records": len(df),
                        "columns": len(df.columns),
                        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
                    }
                    
                    # Get last modified time for features.csv
                    if file_path.name == "features.csv":
                        summary["last_updated"] = datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat()
                        
                except Exception:
                    processed_files[file_path.name] = {"error": "Could not read file"}
            
            summary["processed_data"] = processed_files
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get data summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data summary: {str(e)}")

@router.post("/data/generate")
async def generate_new_data(
    n_days: int = Field(7, ge=1, le=30),
    background_tasks: BackgroundTasks = None
):
    """Generate new synthetic data"""
    try:
        if background_tasks:
            background_tasks.add_task(generate_data_task, n_days)
            
            return {
                "message": f"Data generation started for {n_days} days",
                "status": "started",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Synchronous generation for small datasets
            from src.data.data_generator import SystemDataGenerator
            
            generator = SystemDataGenerator()
            metrics_df, logs_df = generator.generate_all_data(n_days=n_days)
            
            return {
                "message": "Data generation completed",
                "metrics_records": len(metrics_df),
                "log_records": len(logs_df),
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

# Alerting endpoints
@router.get("/alerts", response_model=List[SystemAlert])
async def get_system_alerts():
    """Get current system alerts"""
    try:
        # Mock alerts for demonstration
        # In production, this would read from a database or alert system
        
        alerts = []
        
        # Check model performance
        try:
            performance_data = await get_model_performance()
            for perf in performance_data:
                if perf.roc_auc < 0.7:
                    alert = SystemAlert(
                        alert_id=f"model_performance_{perf.model_name}",
                        severity="MEDIUM",
                        message=f"Model {perf.model_name} performance degraded (AUC: {perf.roc_auc:.3f})",
                        timestamp=datetime.now().isoformat(),
                        resolved=False
                    )
                    alerts.append(alert)
        except Exception:
            pass
        
        # Check data freshness
        try:
            data_summary = await get_data_summary()
            if data_summary.get("last_updated"):
                last_update = datetime.fromisoformat(data_summary["last_updated"])
                hours_old = (datetime.now() - last_update).total_seconds() / 3600
                
                if hours_old > 24:
                    alert = SystemAlert(
                        alert_id="data_freshness",
                        severity="LOW",
                        message=f"Training data is {hours_old:.1f} hours old",
                        timestamp=datetime.now().isoformat(),
                        resolved=False
                    )
                    alerts.append(alert)
        except Exception:
            pass
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

# Utility endpoints
@router.get("/system/status")
async def get_system_status():
    """Get overall system health status"""
    try:
        status = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check models
        try:
            model_status = await get_model_status()
            status["components"]["models"] = {
                "status": "healthy" if model_status["total_models"] > 0 else "warning",
                "details": f"{model_status['total_models']} models loaded"
            }
        except Exception as e:
            status["components"]["models"] = {
                "status": "error",
                "details": str(e)
            }
        
        # Check data
        try:
            data_summary = await get_data_summary()
            has_processed_data = bool(data_summary.get("processed_data"))
            status["components"]["data"] = {
                "status": "healthy" if has_processed_data else "warning",
                "details": "Processed data available" if has_processed_data else "No processed data"
            }
        except Exception as e:
            status["components"]["data"] = {
                "status": "error",
                "details": str(e)
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in status["components"].values()]
        if "error" in component_statuses:
            status["overall_status"] = "error"
        elif "warning" in component_statuses:
            status["overall_status"] = "warning"
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# Background tasks
async def retrain_models_task(model_types: Optional[List[str]], use_latest_data: bool):
    """Background task for model retraining"""
    try:
        logger.info("Starting model retraining...")
        
        # Import here to avoid circular imports
        from src.models.training import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Check if we need to generate new features first
        if use_latest_data:
            from src.features.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            engineer.process_features()
        
        # Train models
        results, best_model_name, best_model = trainer.train_all_models(use_grid_search=False)
        
        logger.info(f"Model retraining completed. Best model: {best_model_name}")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")

async def generate_data_task(n_days: int):
    """Background task for data generation"""
    try:
        logger.info(f"Starting data generation for {n_days} days...")
        
        from src.data.data_generator import SystemDataGenerator
        
        generator = SystemDataGenerator()
        metrics_df, logs_df = generator.generate_all_data(n_days=n_days)
        
        logger.info(f"Data generation completed: {len(metrics_df)} metrics, {len(logs_df)} logs")
        
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")

# Export router
__all__ = ['router']