from app import db
from datetime import datetime
import json

class EnvironmentalData(db.Model):
    """Model for storing environmental data"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    pressure = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    wind_direction = db.Column(db.Float, nullable=False)
    visibility = db.Column(db.Float, nullable=True)
    uv_index = db.Column(db.Float, nullable=True)
    air_quality_index = db.Column(db.Float, nullable=True)
    weather_condition = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(100), nullable=False, default='Default Location')
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'visibility': self.visibility,
            'uv_index': self.uv_index,
            'air_quality_index': self.air_quality_index,
            'weather_condition': self.weather_condition,
            'location': self.location
        }

class MLModel(db.Model):
    """Model for storing ML model metadata"""
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    accuracy_score = db.Column(db.Float, nullable=True)
    precision_score = db.Column(db.Float, nullable=True)
    recall_score = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    training_data_size = db.Column(db.Integer, nullable=True)
    feature_importance = db.Column(db.Text, nullable=True)  # JSON string
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'accuracy_score': self.accuracy_score,
            'precision_score': self.precision_score,
            'recall_score': self.recall_score,
            'f1_score': self.f1_score,
            'training_data_size': self.training_data_size,
            'feature_importance': json.loads(self.feature_importance) if self.feature_importance else None,
            'is_active': self.is_active
        }

class Prediction(db.Model):
    """Model for storing predictions"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False)
    predicted_temperature = db.Column(db.Float, nullable=True)
    predicted_humidity = db.Column(db.Float, nullable=True)
    predicted_pressure = db.Column(db.Float, nullable=True)
    predicted_weather_condition = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    input_features = db.Column(db.Text, nullable=True)  # JSON string
    model_id = db.Column(db.Integer, db.ForeignKey('ml_model.id'), nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'predicted_temperature': self.predicted_temperature,
            'predicted_humidity': self.predicted_humidity,
            'predicted_pressure': self.predicted_pressure,
            'predicted_weather_condition': self.predicted_weather_condition,
            'confidence_score': self.confidence_score,
            'input_features': json.loads(self.input_features) if self.input_features else None,
            'model_id': self.model_id
        }
