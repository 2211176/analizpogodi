import requests
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression

# Замените YOUR_API_KEY на ваш API ключ от WeatherAPI
API_KEY = "cd380213c715482f9b6140006241303"
city = "Kazan"
data = []

for i in range(10):
    date = datetime.now() - timedelta(days=i+1)
    url = f"https://api.weatherapi.com/v1/history.json?key={API_KEY}&q={city}&dt={date.strftime('%Y-%m-%d')}"
    
    response = requests.get(url)
    weather_data = response.json()
    
    if 'forecast' in weather_data:
        temperature = weather_data['forecast']['forecastday'][0]['day']['avgtemp_c']
        humidity = weather_data['forecast']['forecastday'][0]['day']['avghumidity']
        precipitation = weather_data['forecast']['forecastday'][0]['day']['totalprecip_mm']
        wind_speed = weather_data['forecast']['forecastday'][0]['day']['maxwind_kph']
        
        data.append({'date': date.strftime('%Y-%m-%d'), 'temperature': temperature, 'humidity': humidity, 'precipitation': precipitation, 'wind_speed': wind_speed})
    else:
        print(f"Ошибка при получении данных за {date.strftime('%Y-%m-%d')}.")

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['day_number'] = (df['date'] - df['date'].min()).dt.days

X = df[['day_number', 'humidity', 'precipitation', 'wind_speed']]
y = df['temperature']

model = LinearRegression()
model.fit(X, y)

next_day_number = df['day_number'].max() + 1

url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&days=1"
response = requests.get(url)
weather_data = response.json()

if 'forecast' in weather_data:
    next_humidity = weather_data['forecast']['forecastday'][0]['day']['avghumidity']
    next_precipitation = weather_data['forecast']['forecastday'][0]['day']['totalprecip_mm']
    next_wind_speed = weather_data['forecast']['forecastday'][0]['day']['maxwind_kph']
    
    next_temperature = model.predict([[next_day_number, next_humidity, next_precipitation, next_wind_speed]])[0]
    
    print(f"Предполагаемая температура на следующий день: {next_temperature}°C")
else:
    print("Ошибка при получении данных о погоде на следующий день.")