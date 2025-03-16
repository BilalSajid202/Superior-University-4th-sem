from flask import Flask, render_template, request, jsonify
import requests
import os
from datetime import datetime
import logging
import json
from functools import wraps
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# WeatherAPI.com API key
API_KEY = "5e0e290a3ee841a387594411251603"
BASE_URL = "https://api.weatherapi.com/v1"
CURRENT_WEATHER_URL = f"{BASE_URL}/current.json"
FORECAST_WEATHER_URL = f"{BASE_URL}/forecast.json"

# Simple in-memory cache
cache = {}
CACHE_TIMEOUT = 600  # 10 minutes in seconds

def cached(timeout=CACHE_TIMEOUT):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create a cache key from the function name and arguments
            cache_key = f.__name__ + str(args) + str(kwargs)
            
            # Check if we have a valid cached response
            if cache_key in cache:
                cached_result, timestamp = cache[cache_key]
                if time.time() - timestamp < timeout:
                    app.logger.info(f"Cache hit for {cache_key}")
                    return cached_result
            
            # If not in cache or expired, call the original function
            result = f(*args, **kwargs)
            
            # Store the result in cache
            cache[cache_key] = (result, time.time())
            app.logger.info(f"Cache miss for {cache_key}, stored result")
            
            return result
        return decorated_function
    return decorator

@app.route('/')
def home():
    return render_template('index.html')

@cached()
def get_weather_data(endpoint, params):
    """Fetch weather data from the API with caching"""
    try:
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            app.logger.error(f"API returned status code {response.status_code}: {response.text}")
            return None, f"Error fetching weather data. Status code: {response.status_code}"
        
        try:
            data = response.json()
            return data, None
        except requests.exceptions.JSONDecodeError as e:
            app.logger.error(f"Failed to decode JSON: {e}")
            app.logger.error(f"Response content: {response.text}")
            return None, "Error parsing weather data. Please check your API key."
            
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Request error: {e}")
        return None, "Error connecting to weather service. Please try again later."
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return None, "An unexpected error occurred. Please try again."

def process_current_weather(data):
    """Process current weather data"""
    location = data['location']
    current = data['current']
    
    weather_info = {
        'city': location['name'],
        'country': location['country'],
        'region': location['region'],
        'temperature': {
            'c': current['temp_c'],
            'f': current['temp_f']
        },
        'feels_like': {
            'c': current['feelslike_c'],
            'f': current['feelslike_f']
        },
        'condition': current['condition']['text'],
        'icon': current['condition']['icon'],
        'humidity': current['humidity'],
        'pressure': current['pressure_mb'],
        'wind_speed': {
            'kph': current['wind_kph'],
            'mph': current['wind_mph']
        },
        'wind_direction': current['wind_dir'],
        'uv': current['uv'],
        'cloud': current['cloud'],
        'visibility': {
            'km': current['vis_km'],
            'miles': current['vis_miles']
        },
        'last_updated': current['last_updated'],
        'datetime': datetime.now().strftime("%A, %d %B %Y, %I:%M %p"),
        'is_day': current['is_day'] == 1,
        'location': {
            'lat': location['lat'],
            'lon': location['lon'],
            'tz_id': location['tz_id'],
            'localtime': location['localtime']
        }
    }
    
    # Add air quality data if available
    if 'air_quality' in current:
        air_quality = current['air_quality']
        weather_info['air_quality'] = {
            'co': round(air_quality.get('co', 0), 2),
            'no2': round(air_quality.get('no2', 0), 2),
            'o3': round(air_quality.get('o3', 0), 2),
            'pm2_5': round(air_quality.get('pm2_5', 0), 2),
            'pm10': round(air_quality.get('pm10', 0), 2),
            'us_epa_index': air_quality.get('us-epa-index', 0),
            'gb_defra_index': air_quality.get('gb-defra-index', 0)
        }
    
    return weather_info

def process_forecast_data(data, days=3):
    """Process forecast data"""
    forecast_days = data['forecast']['forecastday'][:days]
    processed_forecast = []
    
    for day in forecast_days:
        day_data = {
            'date': day['date'],
            'date_formatted': datetime.strptime(day['date'], '%Y-%m-%d').strftime('%A, %b %d'),
            'max_temp': {
                'c': day['day']['maxtemp_c'],
                'f': day['day']['maxtemp_f']
            },
            'min_temp': {
                'c': day['day']['mintemp_c'],
                'f': day['day']['mintemp_f']
            },
            'avg_temp': {
                'c': day['day']['avgtemp_c'],
                'f': day['day']['avgtemp_f']
            },
            'condition': day['day']['condition']['text'],
            'icon': day['day']['condition']['icon'],
            'max_wind_kph': day['day']['maxwind_kph'],
            'max_wind_mph': day['day']['maxwind_mph'],
            'total_precip_mm': day['day']['totalprecip_mm'],
            'total_precip_in': day['day']['totalprecip_in'],
            'avg_humidity': day['day']['avghumidity'],
            'daily_chance_of_rain': day['day']['daily_chance_of_rain'],
            'daily_chance_of_snow': day['day']['daily_chance_of_snow'],
            'uv': day['day']['uv'],
            'hourly': []
        }
        
        # Add hourly data (every 3 hours to keep it manageable)
        for i in range(0, 24, 3):
            hour = day['hour'][i]
            hour_data = {
                'time': hour['time'].split(' ')[1],
                'temp_c': hour['temp_c'],
                'temp_f': hour['temp_f'],
                'condition': hour['condition']['text'],
                'icon': hour['condition']['icon'],
                'wind_kph': hour['wind_kph'],
                'wind_mph': hour['wind_mph'],
                'wind_dir': hour['wind_dir'],
                'humidity': hour['humidity'],
                'chance_of_rain': hour['chance_of_rain'],
                'chance_of_snow': hour['chance_of_snow']
            }
            day_data['hourly'].append(hour_data)
        
        processed_forecast.append(day_data)
    
    return processed_forecast

@app.route('/weather', methods=['POST'])
def weather():
    city = request.form.get('city')
    
    if not city:
        return render_template('index.html', error="Please enter a city name")
    
    # Get forecast data (includes current weather)
    params = {
        'key': API_KEY,
        'q': city,
        'days': 7,
        'aqi': 'yes',
        'alerts': 'yes'
    }
    
    data, error = get_weather_data(FORECAST_WEATHER_URL, params)
    
    if error:
        return render_template('index.html', error=error)
    
    # Process current weather
    current_weather = process_current_weather(data)
    
    # Process forecast data
    forecast = process_forecast_data(data)
    
    # Check for weather alerts
    alerts = data.get('alerts', {}).get('alert', [])
    
    return render_template(
        'weather.html', 
        weather=current_weather, 
        forecast=forecast, 
        alerts=alerts,
        unit='c'  # Default unit
    )

@app.route('/api/weather', methods=['GET'])
def api_weather():
    """API endpoint for AJAX requests"""
    city = request.args.get('city')
    
    if not city:
        return jsonify({'error': 'City parameter is required'}), 400
    
    # Get forecast data
    params = {
        'key': API_KEY,
        'q': city,
        'days': 7,
        'aqi': 'yes',
        'alerts': 'yes'
    }
    
    data, error = get_weather_data(FORECAST_WEATHER_URL, params)
    
    if error:
        return jsonify({'error': error}), 400
    
    # Process data
    current_weather = process_current_weather(data)
    forecast = process_forecast_data(data)
    alerts = data.get('alerts', {}).get('alert', [])
    
    return jsonify({
        'current': current_weather,
        'forecast': forecast,
        'alerts': alerts
    })

@app.route('/api/geolocation', methods=['POST'])
def geolocation():
    """Handle geolocation data"""
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude are required'}), 400
    
    # Get weather for coordinates
    params = {
        'key': API_KEY,
        'q': f"{lat},{lon}",
        'days': 7,
        'aqi': 'yes',
        'alerts': 'yes'
    }
    
    data, error = get_weather_data(FORECAST_WEATHER_URL, params)
    
    if error:
        return jsonify({'error': error}), 400
    
    # Process data
    current_weather = process_current_weather(data)
    forecast = process_forecast_data(data)
    alerts = data.get('alerts', {}).get('alert', [])
    
    return jsonify({
        'current': current_weather,
        'forecast': forecast,
        'alerts': alerts
    })

@app.route('/maps')
def maps():
    """Weather maps page"""
    return render_template('maps.html')

if __name__ == '__main__':
    app.run(debug=True)