"""
FastAPI application for MLOps Error Prediction System.
File: src/api/app.py
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.prediction import PredictionEngine, SystemMetrics, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MLOps Error Prediction API",
    description="Predict system failures before they happen",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global prediction engine
prediction_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction engine on startup"""
    global prediction_engine
    try:
        prediction_engine = PredictionEngine()
        logger.info("✅ Prediction engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize prediction engine: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Error Prediction API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global prediction_engine
    
    return {
        "status": "healthy" if prediction_engine else "unhealthy",
        "models_loaded": list(prediction_engine.models.keys()) if prediction_engine else [],
        "feature_count": len(prediction_engine.feature_columns) if prediction_engine else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(metrics: SystemMetrics):
    """Predict system failure probability"""
    global prediction_engine
    
    if not prediction_engine:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized")
    
    try:
        prediction = prediction_engine.predict(metrics)
        logger.info(f"Prediction made: {prediction.failure_probability:.4f} ({prediction.failure_risk})")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    global prediction_engine
    
    if not prediction_engine:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized")
    
    return {
        "models_loaded": list(prediction_engine.models.keys()),
        "feature_count": len(prediction_engine.feature_columns),
        "features": prediction_engine.feature_columns[:20],  # First 20 features
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/batch")
async def predict_batch(metrics_list: list[SystemMetrics]):
    """Batch prediction for multiple metrics"""
    global prediction_engine
    
    if not prediction_engine:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized")
    
    try:
        predictions = []
        for metrics in metrics_list:
            prediction = prediction_engine.predict(metrics)
            predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
