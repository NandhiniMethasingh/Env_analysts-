import os
import logging
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS
CORS(app)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///environmental_monitor.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models and services
    import models
    from ml_service import MLService
    from data_service import DataService
    
    db.create_all()
    
    # Initialize services
    ml_service = MLService()
    data_service = DataService()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    """ML predictions page"""
    return render_template('predictions.html')

@app.route('/api/environmental-data')
def get_environmental_data():
    """Get current environmental data"""
    try:
        # Fetch current weather data
        current_data = data_service.get_current_weather()
        
        # Fetch historical data from database
        historical_data = data_service.get_historical_data(days=7)
        
        return jsonify({
            'success': True,
            'current': current_data,
            'historical': historical_data
        })
    except Exception as e:
        logging.error(f"Error fetching environmental data: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch environmental data. Please check your API configuration.'
        }), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train the Random Forest model"""
    try:
        # Get training data
        training_data = data_service.get_training_data()
        
        if len(training_data) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training. Need at least 10 data points.'
            }), 400
        
        # Train the model
        metrics = ml_service.train_model(training_data)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'message': 'Model trained successfully'
        })
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to train model: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Make environmental predictions"""
    try:
        # Get input data from request
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            }), 400
        
        # Make prediction
        prediction = ml_service.predict(input_data)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to make prediction: {str(e)}'
        }), 500

@app.route('/api/model-metrics')
def get_model_metrics():
    """Get current model performance metrics"""
    try:
        metrics = ml_service.get_model_metrics()
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logging.error(f"Error getting model metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Model not trained yet. Please train the model first.'
        }), 400

@app.route('/api/future-predictions')
def get_future_predictions():
    """Get future environmental predictions"""
    try:
        # Get current data as base for predictions
        current_data = data_service.get_current_weather()
        
        # Generate predictions for next 7 days
        future_predictions = ml_service.predict_future(current_data, days=7)
        
        return jsonify({
            'success': True,
            'predictions': future_predictions
        })
    except Exception as e:
        logging.error(f"Error getting future predictions: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get future predictions: {str(e)}'
        }), 500

@app.route('/api/export-data')
def export_data():
    """Export environmental data with ML insights"""
    try:
        # Get all data
        current_data = data_service.get_current_weather()
        historical_data = data_service.get_historical_data(days=30)
        model_metrics = ml_service.get_model_metrics()
        
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'current_conditions': current_data,
            'historical_data': historical_data,
            'ml_model_performance': model_metrics,
            'summary': {
                'total_records': len(historical_data),
                'data_quality_score': ml_service.calculate_data_quality_score(historical_data)
            }
        }
        
        return jsonify({
            'success': True,
            'data': export_data
        })
    except Exception as e:
        logging.error(f"Error exporting data: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to export data: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
