"""
Main entry point for MLOps Error Prediction System.
File: main.py
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.api.app import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting MLOps Error Prediction API")
    logger.info("🌐 API will be available at: http://localhost:8000")
    logger.info("📖 Documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
