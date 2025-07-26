import requests
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
from app import db
from models import EnvironmentalData
import numpy as np

class DataService:
    """Service for handling environmental data operations"""
    
    def __init__(self):
        self.openweather_api_key = os.environ.get("OPENWEATHER_API_KEY", "demo_key")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.default_city = "London"  # Default city for demo
    
    def get_current_weather(self, city=None):
        """Fetch current weather data from OpenWeather API"""
        city = city or self.default_city
        
        try:
            # Construct API URL
            url = f"{self.base_url}/weather"
            params = {
                'q': city,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract environmental data
                current_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'wind_direction': data.get('wind', {}).get('deg', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'weather_condition': data['weather'][0]['main'],
                    'location': f"{data['name']}, {data['sys']['country']}"
                }
                
                # Store in database
                self._store_environmental_data(current_data)
                
                return current_data
            else:
                # Return fallback data if API fails
                logging.warning(f"OpenWeather API returned status {response.status_code}")
                return self._get_fallback_data()
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching weather data: {str(e)}")
            return self._get_fallback_data()
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self):
        """Return realistic fallback environmental data"""
        # Generate realistic environmental data for demonstration
        base_temp = 20 + np.random.normal(0, 5)  # Around 20°C with variation
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'temperature': round(base_temp, 1),
            'humidity': round(50 + np.random.normal(0, 15), 1),
            'pressure': round(1013 + np.random.normal(0, 20), 1),
            'wind_speed': round(abs(np.random.normal(5, 3)), 1),
            'wind_direction': round(np.random.uniform(0, 360), 1),
            'visibility': round(10 + np.random.normal(0, 2), 1),
            'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Sunny']),
            'location': 'Demo Location'
        }
    
    def _store_environmental_data(self, data):
        """Store environmental data in database"""
        try:
            env_data = EnvironmentalData(
                temperature=data['temperature'],
                humidity=data['humidity'],
                pressure=data['pressure'],
                wind_speed=data['wind_speed'],
                wind_direction=data['wind_direction'],
                visibility=data.get('visibility'),
                weather_condition=data.get('weather_condition'),
                location=data.get('location', 'Unknown')
            )
            
            db.session.add(env_data)
            db.session.commit()
            
        except Exception as e:
            logging.error(f"Error storing environmental data: {str(e)}")
            db.session.rollback()
    
    def get_historical_data(self, days=30):
        """Get historical environmental data from database"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            data = EnvironmentalData.query.filter(
                EnvironmentalData.timestamp >= cutoff_date
            ).order_by(EnvironmentalData.timestamp.desc()).all()
            
            historical_data = [item.to_dict() for item in data]
            
            # If no historical data, generate some sample data for demonstration
            if not historical_data:
                historical_data = self._generate_sample_historical_data(days)
            
            return historical_data
            
        except Exception as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            return self._generate_sample_historical_data(days)
    
    def _generate_sample_historical_data(self, days=30):
        """Generate sample historical data for demonstration"""
        data = []
        base_date = datetime.utcnow()
        
        for i in range(days):
            date = base_date - timedelta(days=i)
            
            # Generate realistic seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            
            sample_data = {
                'id': i + 1,
                'timestamp': date.isoformat(),
                'temperature': round(seasonal_temp + np.random.normal(0, 5), 1),
                'humidity': round(60 + np.random.normal(0, 20), 1),
                'pressure': round(1013 + np.random.normal(0, 15), 1),
                'wind_speed': round(abs(np.random.normal(8, 4)), 1),
                'wind_direction': round(np.random.uniform(0, 360), 1),
                'visibility': round(10 + np.random.normal(0, 3), 1),
                'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Sunny', 'Foggy']),
                'location': 'Sample Location'
            }
            
            # Ensure reasonable ranges
            sample_data['humidity'] = max(0, min(100, sample_data['humidity']))
            sample_data['visibility'] = max(0.1, sample_data['visibility'])
            
            data.append(sample_data)
        
        return data
    
    def get_training_data(self):
        """Get data suitable for ML model training"""
        try:
            # Get all available data
            all_data = EnvironmentalData.query.all()
            training_data = [item.to_dict() for item in all_data]
            
            # If insufficient real data, supplement with generated data
            if len(training_data) < 100:
                logging.info("Insufficient real data for training, generating additional samples")
                generated_data = self._generate_comprehensive_training_data(200 - len(training_data))
                training_data.extend(generated_data)
            
            return training_data
            
        except Exception as e:
            logging.error(f"Error getting training data: {str(e)}")
            # Return generated training data as fallback
            return self._generate_comprehensive_training_data(200)
    
    def _generate_comprehensive_training_data(self, num_samples=200):
        """Generate comprehensive training data with realistic patterns"""
        data = []
        base_date = datetime.utcnow()
        
        for i in range(num_samples):
            # Create varied temporal patterns
            date = base_date - timedelta(days=np.random.randint(0, 365))
            hour = np.random.randint(0, 24)
            
            # Seasonal and diurnal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
            diurnal_factor = np.sin(2 * np.pi * hour / 24)
            
            # Base temperature with seasonal and daily variation
            base_temp = 15 + 15 * seasonal_factor + 5 * diurnal_factor
            temperature = base_temp + np.random.normal(0, 3)
            
            # Humidity inversely related to temperature
            humidity = 70 - 0.8 * (temperature - 15) + np.random.normal(0, 10)
            humidity = max(10, min(100, humidity))
            
            # Pressure with weather system patterns
            pressure = 1013 + np.random.normal(0, 20)
            
            # Wind speed with some correlation to pressure differences
            wind_speed = abs(5 + (1013 - pressure) * 0.2 + np.random.normal(0, 3))
            
            # Weather conditions based on humidity and temperature
            if humidity > 85:
                weather_condition = np.random.choice(['Rainy', 'Foggy'], p=[0.7, 0.3])
            elif humidity > 70:
                weather_condition = 'Cloudy'
            elif temperature > 25:
                weather_condition = np.random.choice(['Sunny', 'Clear'], p=[0.6, 0.4])
            else:
                weather_condition = 'Clear'
            
            sample_data = {
                'id': i + 1000,  # Offset to avoid conflicts
                'timestamp': date.replace(hour=hour).isoformat(),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'wind_direction': round(np.random.uniform(0, 360), 1),
                'visibility': round(max(0.5, 15 - humidity * 0.1 + np.random.normal(0, 2)), 1),
                'weather_condition': weather_condition,
                'location': 'Training Data Location'
            }
            
            data.append(sample_data)
        
        return data
