import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import logging
from datetime import datetime, timedelta
import json

class MLService:
    """Service for handling machine learning operations"""
    
    def __init__(self):
        self.temperature_model = None
        self.humidity_model = None
        self.pressure_model = None
        self.weather_condition_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction']
        self.model_metrics = {}
        self.is_trained = False
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            if (os.path.exists('temperature_model.joblib') and 
                os.path.exists('humidity_model.joblib') and 
                os.path.exists('pressure_model.joblib') and
                os.path.exists('weather_condition_model.joblib')):
                
                self.temperature_model = joblib.load('temperature_model.joblib')
                self.humidity_model = joblib.load('humidity_model.joblib')
                self.pressure_model = joblib.load('pressure_model.joblib')
                self.weather_condition_model = joblib.load('weather_condition_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.label_encoder = joblib.load('label_encoder.joblib')
                
                # Load metrics
                if os.path.exists('model_metrics.json'):
                    with open('model_metrics.json', 'r') as f:
                        self.model_metrics = json.load(f)
                
                self.is_trained = True
                logging.info("Models loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load existing models: {str(e)}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            joblib.dump(self.temperature_model, 'temperature_model.joblib')
            joblib.dump(self.humidity_model, 'humidity_model.joblib')
            joblib.dump(self.pressure_model, 'pressure_model.joblib')
            joblib.dump(self.weather_condition_model, 'weather_condition_model.joblib')
            joblib.dump(self.scaler, 'scaler.joblib')
            joblib.dump(self.label_encoder, 'label_encoder.joblib')
            
            # Save metrics
            with open('model_metrics.json', 'w') as f:
                json.dump(self.model_metrics, f)
            
            logging.info("Models saved successfully")
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
    
    def _prepare_features(self, data):
        """Prepare features for training or prediction"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Select feature columns
        feature_data = df[self.feature_columns].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())
        
        # Add engineered features
        feature_data['temp_humidity_ratio'] = feature_data['temperature'] / (feature_data['humidity'] + 1e-6)
        feature_data['pressure_normalized'] = (feature_data['pressure'] - 1013.25) / 100
        feature_data['wind_power'] = feature_data['wind_speed'] ** 2
        
        return feature_data
    
    def train_model(self, training_data):
        """Train Random Forest models"""
        try:
            # Convert to DataFrame if needed
            if isinstance(training_data, list):
                df = pd.DataFrame(training_data)
            else:
                df = training_data.copy()
            
            # Prepare features
            X = self._prepare_features(df)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Prepare targets
            y_temp = df['temperature']
            y_humidity = df['humidity']
            y_pressure = df['pressure']
            
            # Handle weather condition encoding
            if 'weather_condition' in df.columns:
                weather_conditions = df['weather_condition'].fillna('Clear')
                y_weather = self.label_encoder.fit_transform(weather_conditions)
            else:
                # Create synthetic weather conditions based on temperature and humidity
                y_weather = self._create_weather_conditions(df['temperature'], df['humidity'])
                y_weather = self.label_encoder.fit_transform(y_weather)
            
            # Split data
            X_train, X_test, y_temp_train, y_temp_test = train_test_split(
                X_scaled, y_temp, test_size=0.2, random_state=42
            )
            _, _, y_humidity_train, y_humidity_test = train_test_split(
                X_scaled, y_humidity, test_size=0.2, random_state=42
            )
            _, _, y_pressure_train, y_pressure_test = train_test_split(
                X_scaled, y_pressure, test_size=0.2, random_state=42
            )
            _, _, y_weather_train, y_weather_test = train_test_split(
                X_scaled, y_weather, test_size=0.2, random_state=42
            )
            
            # Train models
            self.temperature_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.pressure_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.weather_condition_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            self.temperature_model.fit(X_train, y_temp_train)
            self.humidity_model.fit(X_train, y_humidity_train)
            self.pressure_model.fit(X_train, y_pressure_train)
            self.weather_condition_model.fit(X_train, y_weather_train)
            
            # Calculate metrics
            temp_pred = self.temperature_model.predict(X_test)
            humidity_pred = self.humidity_model.predict(X_test)
            pressure_pred = self.pressure_model.predict(X_test)
            weather_pred = self.weather_condition_model.predict(X_test)
            
            # Store metrics
            self.model_metrics = {
                'training_date': datetime.utcnow().isoformat(),
                'training_data_size': len(df),
                'temperature_model': {
                    'r2_score': float(r2_score(y_temp_test, temp_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_temp_test, temp_pred))),
                    'feature_importance': self.temperature_model.feature_importances_.tolist()
                },
                'humidity_model': {
                    'r2_score': float(r2_score(y_humidity_test, humidity_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_humidity_test, humidity_pred))),
                    'feature_importance': self.humidity_model.feature_importances_.tolist()
                },
                'pressure_model': {
                    'r2_score': float(r2_score(y_pressure_test, pressure_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_pressure_test, pressure_pred))),
                    'feature_importance': self.pressure_model.feature_importances_.tolist()
                },
                'weather_condition_model': {
                    'accuracy': float(accuracy_score(y_weather_test, weather_pred)),
                    'precision': float(precision_score(y_weather_test, weather_pred, average='weighted')),
                    'recall': float(recall_score(y_weather_test, weather_pred, average='weighted')),
                    'f1_score': float(f1_score(y_weather_test, weather_pred, average='weighted')),
                    'feature_importance': self.weather_condition_model.feature_importances_.tolist()
                }
            }
            
            self.is_trained = True
            self._save_models()
            
            return self.model_metrics
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise e
    
    def _create_weather_conditions(self, temperature, humidity):
        """Create synthetic weather conditions based on temperature and humidity"""
        conditions = []
        for temp, hum in zip(temperature, humidity):
            if temp > 30 and hum > 80:
                conditions.append('Humid')
            elif temp > 25 and hum < 30:
                conditions.append('Hot')
            elif temp < 5:
                conditions.append('Cold')
            elif hum > 90:
                conditions.append('Foggy')
            elif hum > 70:
                conditions.append('Cloudy')
            else:
                conditions.append('Clear')
        return conditions
    
    def predict(self, input_data):
        """Make predictions using trained models"""
        if not self.is_trained:
            raise ValueError("Models not trained. Please train the models first.")
        
        try:
            # Prepare input features
            if isinstance(input_data, dict):
                # Single prediction
                feature_dict = {col: input_data.get(col, 0) for col in self.feature_columns}
                df = pd.DataFrame([feature_dict])
            else:
                df = pd.DataFrame(input_data)
            
            X = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            temp_pred = self.temperature_model.predict(X_scaled)[0]
            humidity_pred = self.humidity_model.predict(X_scaled)[0]
            pressure_pred = self.pressure_model.predict(X_scaled)[0]
            weather_pred_encoded = self.weather_condition_model.predict(X_scaled)[0]
            weather_pred = self.label_encoder.inverse_transform([weather_pred_encoded])[0]
            
            # Calculate confidence scores
            temp_confidence = np.mean(self.temperature_model.predict(X_scaled))
            weather_confidence = np.max(self.weather_condition_model.predict_proba(X_scaled)[0])
            
            return {
                'predicted_temperature': float(temp_pred),
                'predicted_humidity': float(humidity_pred),
                'predicted_pressure': float(pressure_pred),
                'predicted_weather_condition': weather_pred,
                'confidence_scores': {
                    'temperature': float(temp_confidence),
                    'weather': float(weather_confidence)
                }
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise e
    
    def predict_future(self, current_data, days=7):
        """Generate future predictions"""
        if not self.is_trained:
            raise ValueError("Models not trained. Please train the models first.")
        
        predictions = []
        base_date = datetime.utcnow()
        
        try:
            for day in range(1, days + 1):
                # Use current data as base and add some variation
                future_input = current_data.copy()
                
                # Add temporal variations (simplified approach)
                temp_variation = np.sin(day * np.pi / 365) * 5  # Seasonal variation
                future_input['temperature'] = current_data['temperature'] + temp_variation
                future_input['humidity'] = max(0, min(100, current_data['humidity'] + np.random.normal(0, 5)))
                future_input['pressure'] = current_data['pressure'] + np.random.normal(0, 10)
                
                prediction = self.predict(future_input)
                prediction['prediction_date'] = (base_date + timedelta(days=day)).isoformat()
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error generating future predictions: {str(e)}")
            raise e
    
    def get_model_metrics(self):
        """Get current model performance metrics"""
        if not self.is_trained:
            raise ValueError("Models not trained. No metrics available.")
        return self.model_metrics
    
    def calculate_data_quality_score(self, data):
        """Calculate data quality score"""
        if not data:
            return 0.0
        
        try:
            df = pd.DataFrame(data)
            
            # Calculate completeness
            completeness = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            
            # Calculate consistency (no negative humidity, reasonable temperature ranges)
            consistency_score = 1.0
            if 'humidity' in df.columns:
                invalid_humidity = ((df['humidity'] < 0) | (df['humidity'] > 100)).sum()
                consistency_score -= invalid_humidity / len(df) * 0.3
            
            if 'temperature' in df.columns:
                invalid_temp = ((df['temperature'] < -50) | (df['temperature'] > 60)).sum()
                consistency_score -= invalid_temp / len(df) * 0.3
            
            # Overall score
            quality_score = (completeness * 0.7 + max(0, consistency_score) * 0.3)
            return float(min(1.0, max(0.0, quality_score)))
            
        except Exception as e:
            logging.error(f"Error calculating data quality score: {str(e)}")
            return 0.5  # Default moderate score
