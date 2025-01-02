import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import plotly.graph_objects as go
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Fonction pour récupérer les données historiques d'une crypto-monnaie depuis CoinGecko
def fetch_data(symbol, start=None, end=None):
    cg = CoinGeckoAPI()
    
    # Obtenir l'ID CoinGecko correspondant au symbole
    coin_list = cg.get_coins_list()
    coin_id = None
    for coin in coin_list:
        if coin['id'] == symbol.lower() or coin['symbol'] == symbol.lower():
            coin_id = coin['id']
            break
    if not coin_id:
        raise Exception(f"Symbole '{symbol}' non trouvé sur CoinGecko.")
    
    # Convertir les dates en timestamp UNIX
    if end:
        end_datetime = datetime.strptime(end, '%Y-%m-%d')
    else:
        end_datetime = datetime.now()
    
    if start:
        start_datetime = datetime.strptime(start, '%Y-%m-%d')
        # Calculer la différence en jours
        delta = end_datetime - start_datetime
        if delta.days > 365:
            # Limiter à 365 jours
            start_datetime = end_datetime - timedelta(days=365)
            print(f"La plage de dates a été ajustée à 365 jours : {start_datetime.date()} - {end_datetime.date()}")
    else:
        # Si start n'est pas fourni, récupérer les 365 derniers jours
        start_datetime = end_datetime - timedelta(days=365)
    
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())
    
    # Récupérer les données historiques
    data = cg.get_coin_market_chart_range_by_id(id=coin_id, vs_currency='usd',
                                               from_timestamp=start_timestamp,
                                               to_timestamp=end_timestamp)
    
    if 'prices' not in data:
        if 'error' in data:
            raise Exception(data['error'])
        else:
            raise Exception("Impossible de récupérer les données historiques.")
    
    # Transformer les données en DataFrame
    df = pd.DataFrame(data['prices'], columns=['Timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[['Date', 'Close']]
    df.sort_values('Date', inplace=True)
    
    # Filtrer par dates si fournies (déjà limité à 365 jours)
    return df.reset_index(drop=True)

# Ajout d'indicateurs techniques simplifiés
def add_technical_indicators(data):
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['SMA'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    data['MACD'] = MACD(data['Close']).macd()
    data.dropna(inplace=True)
    return data

# Préparation des données
def prepare_data(data, look_back, future_horizon):
    features = data.columns.tolist()
    features.remove('Date')
    features.remove('Close')
    for i in range(1, future_horizon + 1):
        data[f'Future_{i}'] = data['Close'].shift(-i)
    data.dropna(inplace=True)

    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(train_data[features])
    scaler_y.fit(train_data[[f'Future_{i}' for i in range(1, future_horizon + 1)]])

    X_train = scaler_X.transform(train_data[features])
    Y_train = scaler_y.transform(train_data[[f'Future_{i}' for i in range(1, future_horizon + 1)]])
    X_test = scaler_X.transform(test_data[features])
    Y_test = scaler_y.transform(test_data[[f'Future_{i}' for i in range(1, future_horizon + 1)]])

    def create_sequences(X, Y, look_back):
        Xs, Ys = [], []
        for i in range(len(X) - look_back):
            Xs.append(X[i:i + look_back])
            Ys.append(Y[i + look_back - 1])
        return np.array(Xs), np.array(Ys)

    X_train, Y_train = create_sequences(X_train, Y_train, look_back)
    X_test, Y_test = create_sequences(X_test, Y_test, look_back)

    dates_test = test_data['Date'].iloc[look_back - 1 + future_horizon:].reset_index(drop=True)
    return X_train, Y_train, X_test, Y_test, scaler_X, scaler_y, dates_test, data, features

# Construction du modèle simplifié avec régularisation L2
def build_model(look_back, n_features, future_horizon):
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_shape=(look_back, n_features),
                   kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.7))
    model.add(LSTM(512, kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.3))
    model.add(Dense(future_horizon))
    model.compile(optimizer='adam', loss='mae')
    return model

# Fonctions de visualisation
def plot_example_prediction(Y_test, predictions, scaler_y, dates_test):
    inv_Y_test = scaler_y.inverse_transform(Y_test)
    inv_preds = scaler_y.inverse_transform(predictions)
    example_idx = -1
    fh = Y_test.shape[1]
    example_dates = pd.date_range(start=dates_test.iloc[example_idx], periods=fh, freq='D')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=example_dates, y=inv_Y_test[example_idx], mode='lines+markers', name='Vrai'))
    fig.add_trace(go.Scatter(x=example_dates, y=inv_preds[example_idx], mode='lines+markers', name='Prédit'))
    fig.update_layout(title='Exemple de prévision multi-step (dans la zone de test)',
                      xaxis_title='Date',
                      yaxis_title='Prix')
    fig.show()

def plot_future_forecast(data, model, scaler_X, scaler_y, features, look_back, future_horizon):
    last_window = data[features].iloc[-look_back:]
    scaled_last_window = scaler_X.transform(last_window)
    scaled_last_window = scaled_last_window.reshape(1, look_back, len(features))
    future_preds = model.predict(scaled_last_window)
    inv_future_preds = scaler_y.inverse_transform(future_preds).flatten()
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_horizon, freq='D')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historique'))
    fig.add_trace(go.Scatter(x=future_dates, y=inv_future_preds, mode='lines+markers', name='Prédictions Futures', line=dict(color='red')))
    fig.update_layout(title='Prévisions Futures au-delà des données disponibles',
                      xaxis_title='Date',
                      yaxis_title='Prix')
    fig.show()

if __name__ == "__main__":
    # Définir les paramètres
    symbol = 'solana'  # Exemple : 'ethereum' pour Ethereum, 'solana' pour Solana
    end_date = '2024-12-13'
    look_back = 40
    future_horizon = 10

    try:
        # Calcul automatique de la date de début (365 jours avant la date de fin)
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        start_datetime = end_datetime - timedelta(days=365)
        start_date = start_datetime.strftime('%Y-%m-%d')

        # Récupérer les données
        data = fetch_data(symbol, start_date, end_date)
        print("Données récupérées avec succès :")
        print(data.head())

        # Ajout des indicateurs techniques
        data = add_technical_indicators(data)

        # Préparation des données pour l'entraînement et le test
        X_train, Y_train, X_test, Y_test, scaler_X, scaler_y, dates_test, full_data, features = prepare_data(
            data, look_back, future_horizon
        )

        # Construction du modèle
        model = build_model(look_back, X_train.shape[2], future_horizon)

        # Définition des callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        # Entraînement du modèle avec sauvegarde de l'historique
        print("Entraînement du modèle...")
        history = model.fit(
            X_train, Y_train,
            epochs=150,
            batch_size=64,
            validation_data=(X_test, Y_test),
            callbacks=[reduce_lr, early_stopping]
        )

        # Tracer les courbes de perte pour évaluer l'overfitting
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Perte Entraînement')
        plt.plot(history.history['val_loss'], label='Perte Validation')
        plt.title('Courbes de Perte')
        plt.xlabel('Épochs')
        plt.ylabel('Perte (MAE)')
        plt.legend()
        plt.show()

        # Prédictions sur les données de test
        predictions = model.predict(X_test)

        # Évaluation des performances
        inv_Y_test = scaler_y.inverse_transform(Y_test)
        inv_preds = scaler_y.inverse_transform(predictions)
        mae = mean_absolute_error(inv_Y_test[:, 0], inv_preds[:, 0])
        rmse = np.sqrt(mean_squared_error(inv_Y_test[:, 0], inv_preds[:, 0]))
        print(f"Mean Absolute Error (Day 1): {mae}")
        print(f"Root Mean Squared Error (Day 1): {rmse}")

        # Visualisation des prédictions sur les données de test
        plot_example_prediction(Y_test, predictions, scaler_y, dates_test)

        # Prédictions futures au-delà des données disponibles
        plot_future_forecast(full_data, model, scaler_X, scaler_y, features, look_back, future_horizon)

    except Exception as e:
        print(f"Erreur : {e}")
