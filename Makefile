# MLOps Error Prediction Makefile

.PHONY: help install setup data train api test clean docker all

# Default target
help:
	@echo "🚀 MLOps Error Prediction System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup project structure"
	@echo "  make data        - Generate sample data"
	@echo "  make features    - Generate features"
	@echo "  make train       - Train models"
	@echo "  make api         - Start API server"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean generated files"
	@echo "  make docker      - Build and run with Docker"
	@echo "  make all         - Run complete pipeline"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Setup project structure
setup:
	@echo "🏗️ Setting up project structure..."
	python scripts/setup.py
	cp .env.example .env
	@echo "✅ Project setup complete"

# Generate sample data
data:
	@echo "📊 Generating sample data..."
	python src/data/data_generator.py
	@echo "✅ Sample data generated"

# Feature engineering
features:
	@echo "🔧 Creating features..."
	python src/features/feature_engineering.py
	@echo "✅ Features created"

# Train models
train:
	@echo "🤖 Training models..."
	python src/models/training.py
	@echo "✅ Models trained"

# Start API server
api:
	@echo "🌐 Starting API server..."
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests completed"

# Format code
format:
	@echo "🎨 Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	@echo "✅ Code formatted"

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 src/ tests/ scripts/
	@echo "✅ Code linted"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf logs/*
	rm -rf mlruns/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Docker operations
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -f docker/Dockerfile -t mlops-error-prediction .

docker-run:
	@echo "🐳 Running Docker container..."
	docker-compose -f docker/docker-compose.yml up -d

docker: docker-build docker-run

# Full pipeline
pipeline: data features train
	@echo "🎉 Complete pipeline executed!"

# Development setup
dev-setup: install setup data features train
	@echo "🎉 Development environment ready!"

# Production setup
prod-setup: install setup
	@echo "🎉 Production environment ready!"

# Quick test of the system
quick-test: data features train
	@echo "🚀 Testing API..."
	python -c "import requests; print('API Test:', requests.post('http://localhost:8000/predict', json={'cpu_usage': 85, 'memory_usage': 90, 'disk_usage': 45, 'network_latency_ms': 120, 'error_count': 3, 'response_time_ms': 450, 'active_connections': 75}).status_code)"

# All-in-one command
all: install setup data features train
	@echo "🎉 MLOps Error Prediction System is ready!"
	@echo "🌐 Start the API with: make api"
	@echo "🧪 Run tests with: make test"

# Help with getting started
getting-started:
	@echo "🚀 Getting Started with MLOps Error Prediction"
	@echo ""
	@echo "1. First time setup:"
	@echo "   make dev-setup"
	@echo ""
	@echo "2. Start the API:"
	@echo "   make api"
	@echo ""
	@echo "3. Test the system:"
	@echo "   make test"
	@echo ""
	@echo "4. View documentation:"
	@echo "   Open http://localhost:8000/docs in your browser"