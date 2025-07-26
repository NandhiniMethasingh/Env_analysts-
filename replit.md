# Environment Monitor Hub

## Overview

This is a Flask-based environmental monitoring application that combines real-time weather data collection with machine learning predictions. The system fetches environmental data from OpenWeather API, stores it in a database, and uses Random Forest models to make predictions about future environmental conditions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Flask Web Framework**: Main application framework providing REST APIs and web routes
- **SQLAlchemy ORM**: Database abstraction layer using DeclarativeBase for model definitions
- **Service Layer Pattern**: Business logic separated into dedicated services (MLService, DataService)
- **CORS Enabled**: Cross-origin resource sharing for frontend-backend communication

### Frontend Architecture
- **Server-Side Rendered Templates**: Jinja2 templates for HTML generation
- **Bootstrap 5**: Dark theme UI framework for responsive design
- **Chart.js**: Data visualization library for environmental charts
- **Vanilla JavaScript**: Client-side interaction with ES6 classes for dashboard and predictions

### Database Design
- **SQLite/PostgreSQL**: Configurable database backend via environment variables
- **Two Main Models**: 
  - `EnvironmentalData`: Stores sensor readings and weather data
  - `MLModel`: Stores machine learning model metadata and performance metrics

## Key Components

### Data Collection Service (DataService)
- **External API Integration**: OpenWeather API for real-time weather data
- **Data Normalization**: Converts API responses to standardized environmental metrics
- **Automatic Storage**: Persists collected data to database for historical analysis

### Machine Learning Service (MLService)
- **Multiple Prediction Models**: Separate Random Forest models for temperature, humidity, pressure, and weather conditions
- **Model Persistence**: Saves/loads trained models using joblib serialization
- **Performance Tracking**: Stores accuracy metrics and feature importance data
- **Preprocessing Pipeline**: StandardScaler for numerical features, LabelEncoder for categorical data

### Web Interface
- **Dashboard Page**: Real-time environmental conditions with data visualization
- **Predictions Page**: Interactive ML prediction interface with form inputs
- **Responsive Design**: Mobile-friendly layout using Bootstrap grid system

## Data Flow

1. **Data Collection**: DataService fetches weather data from OpenWeather API
2. **Data Storage**: Environmental readings stored in EnvironmentalData table
3. **Model Training**: MLService uses historical data to train Random Forest models
4. **Prediction Generation**: Trained models predict future environmental conditions
5. **Visualization**: Frontend displays current conditions and predictions via charts
6. **User Interaction**: Web interface allows manual prediction inputs and model retraining

## External Dependencies

### APIs
- **OpenWeather API**: Primary source for environmental data (temperature, humidity, pressure, wind)
- **API Key Management**: Configurable via OPENWEATHER_API_KEY environment variable

### Python Libraries
- **Flask Ecosystem**: flask, flask-sqlalchemy, flask-cors for web framework
- **Machine Learning**: scikit-learn for Random Forest models, pandas/numpy for data processing
- **Utilities**: requests for HTTP calls, joblib for model serialization

### Frontend Libraries
- **Bootstrap 5**: UI framework with dark theme
- **Chart.js**: Data visualization and charting
- **Font Awesome**: Icon library for UI elements

## Deployment Strategy

### Environment Configuration
- **Database URL**: Configurable via DATABASE_URL environment variable (defaults to SQLite)
- **Session Security**: SESSION_SECRET environment variable for Flask sessions
- **API Keys**: External service credentials via environment variables

### Database Management
- **Auto-Migration**: SQLAlchemy creates tables automatically on startup
- **Connection Pooling**: Configured with pool_recycle and pool_pre_ping for reliability
- **Cross-Database Support**: Works with SQLite for development, PostgreSQL for production

### Application Structure
- **Single Entry Point**: main.py runs the Flask application
- **Modular Design**: Separate files for models, services, and static assets
- **Static Asset Serving**: CSS/JS files served directly by Flask in development

### Scaling Considerations
- **Service Separation**: ML and data services can be extracted to separate microservices
- **Database Optimization**: Connection pooling and query optimization built-in
- **Model Persistence**: Trained models saved to disk for consistent predictions across restarts