import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
from sklearn.cluster import KMeans
from clearml import Task
import joblib
import matplotlib.pyplot as plt
import sys
#   streamlit run c:/games/PYTHONPROJECTS/WEBPRIL/base.py
def stop_clearml_experiment():
    task = Task.current_task()
    if task:
        task.close()
        print("Эксперимент в ClearML успешно остановлен")
    else:
        print("Не удалось найти текущий эксперимент в ClearML")
stop_clearml_experiment()
clearml_task = Task.init(project_name="Weather Prediction", task_name="Temperature Prediction")
# WeatherAPI key
API_KEY = "cd380213c715482f9b6140006241303"
city = "Kazan"
data = []

st.title('Weather Prediction')

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
        st.write(f"Error getting data for {date.strftime('%Y-%m-%d')}.")

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df['day_number'] = (df['date'] - df['date'].min()).dt.days
X = df[['day_number', 'humidity', 'precipitation', 'wind_speed']]
y = df['temperature']
cluster_model = KMeans(n_clusters=3, random_state=42)
clusters = cluster_model.fit_predict(X)

xy = st.number_input("На какое количество дней вы хотели бы получить предсказание:", min_value=1, max_value=7, value=1, step=1)

for i in range(1, xy+1):
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

        st.write(f"Predicted temperature for day {i}: {next_temperature}°C")
    else:
        st.write(f"Error getting weather data for day {i}.")

    df = df.iloc[1:]
    new_row = pd.DataFrame({'date': [datetime.now() + timedelta(days=i)], 'temperature': [next_temperature], 'humidity': [next_humidity], 'precipitation': [next_precipitation], 'wind_speed': [next_wind_speed]})
    df = pd.concat([df, new_row], ignore_index=True)
    clearml_task.upload_artifact("weather_prediction.csv", df)
    joblib.dump(cluster_model, "cluster_model.pkl")
    clearml_task.upload_artifact("cluster_model.pkl", "cluster_model.pkl")
    clearml_task.get_logger().report_text(f"Predicted temperature for day {i}: {next_temperature}°C")

# Сортировка данных по дате
df = df.sort_values(by='date')
# Построение графика с темным фоном и яркими линиями
st.write("### График прогноза температуры")
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  # Установка темного фона
plt.plot(df['date'], df['temperature'], color='white', marker='o', linestyle='-')  # Яркие линии белого цвета
plt.xlabel('Дата', color='white')  # Цвет текста на осях
plt.ylabel('Температура (°C)', color='white')
plt.title('Прогноз температуры', color='white')  # Цвет заголовка
plt.xticks(color='white')  # Цвет меток на осях
plt.yticks(color='white')
plt.grid(True)
st.pyplot(plt)
if st.button('Остановить эксперимент'):
    stop_clearml_experiment()