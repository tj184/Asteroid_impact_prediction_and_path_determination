import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import LSTM, Dense
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import datetime

def fetch_nasa_data(start_date, end_date, api_key="DEMO_KEY"):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    all_data = []
    for date in data['near_earth_objects']:
        for obj in data['near_earth_objects'][date]:
            all_data.append({
                'name': obj['name'],
                'diameter': obj['estimated_diameter']['meters']['estimated_diameter_max'],
                'velocity': float(obj['close_approach_data'][0]['relative_velocity']['kilometers_per_hour']),
                'distance': float(obj['close_approach_data'][0]['miss_distance']['kilometers']),
                'hazardous': obj['is_potentially_hazardous_asteroid']
            })
    return pd.DataFrame(all_data)

def train_rf_classifier(df):
    df['hazardous'] = df['hazardous'].astype(int)
    X = df[['diameter', 'velocity', 'distance']]
    y = df['hazardous']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model

def generate_trajectory_data():
    time_steps = 100
    data = np.sin(np.linspace(0, 50, time_steps)) + np.random.normal(0, 0.1, time_steps)
    X, y = [], []
    for i in range(time_steps - 10):
        X.append(data[i:i+10])
        y.append(data[i+10])
    return np.array(X).reshape(-1, 10, 1), np.array(y)

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def monte_carlo_simulation(radius=6371, trials=10000):
    impacts = 0
    for _ in range(trials):
        distance = np.random.normal(15000, 5000)
        if distance <= radius:
            impacts += 1
    return impacts / trials

def plot_asteroid_trajectory():
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='blue', width=4))])
    fig.update_layout(title='Asteroid Trajectory', scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()

if __name__ == '__main__':
    start = datetime.date.today().isoformat()
    end = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    df = fetch_nasa_data(start, end)
    print(df.head())

    rf_model = train_rf_classifier(df)

    X, y = generate_trajectory_data()
    lstm_model = build_lstm_model()
    lstm_model.fit(X, y, epochs=10, verbose=0)
    print("LSTM Model trained")

    impact_prob = monte_carlo_simulation()
    print(f"Estimated impact probability: {impact_prob * 100:.2f}%")

    plot_asteroid_trajectory()
