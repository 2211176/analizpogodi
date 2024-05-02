import requests
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import KMeans
from clearml import Task
import joblib

clearml_task = Task.init(project_name="Weather Prediction", task_name="Temperature Prediction")

# Replace YOUR_API_KEY with your WeatherAPI key
API_KEY = "cd380213c715482f9b6140006241303"
city = "Kazan"
data = []

for i in range(7):
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
        print(f"Error getting data for {date.strftime('%Y-%m-%d')}.")

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['day_number'] = (df['date'] - df['date'].min()).dt.days

X = df[['day_number', 'humidity', 'precipitation', 'wind_speed']]
y = df['temperature']

cluster_model = KMeans(n_clusters=3, random_state=42)
clusters = cluster_model.fit_predict(X)

for i in range(1, 8):
    next_day_number = df['day_number'].max() + i

    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={city}&dt={datetime.now().strftime('%Y-%m-%d')}&days=1"
    response = requests.get(url)
    weather_data = response.json()

    if 'forecast' in weather_data:
        next_humidity = weather_data['forecast']['forecastday'][0]['day']['avghumidity']
        next_precipitation = weather_data['forecast']['forecastday'][0]['day']['totalprecip_mm']
        next_wind_speed = weather_data['forecast']['forecastday'][0]['day']['maxwind_kph']

        next_cluster = cluster_model.predict([[next_day_number, next_humidity, next_precipitation, next_wind_speed]])[0]

        next_temperature = df.loc[clusters == next_cluster, 'temperature'].mean()

        print(f"Predicted temperature for day {i}: {next_temperature}°C")

        # Remove the oldest day and add the first predicted day
        df = df.iloc[1:]
        new_row = pd.DataFrame({'date': [datetime.now() + timedelta(days=i)], 'temperature': [next_temperature], 'humidity': [next_humidity], 'precipitation': [next_precipitation], 'wind_speed': [next_wind_speed]})
        df = pd.concat([df, new_row], ignore_index=True)

        clearml_task.upload_artifact("weather_prediction.csv", df)
        joblib.dump(cluster_model, "cluster_model.pkl")
        clearml_task.upload_artifact("cluster_model.pkl", "cluster_model.pkl")
        clearml_task.get_logger().report_text(f"Predicted temperature for day {i}: {next_temperature}°C")
    else:
        print(f"Error getting weather data for day {i}.")
