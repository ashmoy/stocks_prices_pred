import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

def fetch_data_from_yf(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    
    # Aplatir les colonnes multi-indexées
    df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    # Renommer la colonne "Close ^GSPC" en "Close"
    df.rename(columns={"Close ^GSPC": "Close"}, inplace=True)
    
    return df

# Ajouter des indicateurs techniques
def add_technical_indicators(data):
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['SMA'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    data['MACD'] = MACD(data['Close']).macd()
    data.dropna(inplace=True)
    return data

# Préparation des données
def prepare_data(data, look_back):
    features = data.columns.tolist()
    features.remove('Date')
    features.remove('Close')
    data.dropna(inplace=True)

    split_index = int(len(data) * 0.8)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(train_data[features])
    scaler_y.fit(train_data[['Close']])

    X_train = scaler_X.transform(train_data[features])
    Y_train = scaler_y.transform(train_data[['Close']])
    X_test = scaler_X.transform(test_data[features])
    Y_test = scaler_y.transform(test_data[['Close']])

    def create_sequences(X, Y, look_back):
        Xs, Ys = [], []
        for i in range(len(X) - look_back):
            Xs.append(X[i:i + look_back])
            Ys.append(Y[i + look_back - 1])
        return np.array(Xs), np.array(Ys)

    X_train, Y_train = create_sequences(X_train, Y_train, look_back)
    X_test, Y_test = create_sequences(X_test, Y_test, look_back)

    dates_test = test_data['Date'].iloc[look_back - 1:].reset_index(drop=True)
    return X_train, Y_train, X_test, Y_test, scaler_X, scaler_y, dates_test, data, features

# Construction du modèle
def build_model(look_back, n_features):
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_shape=(look_back, n_features),
                   kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.7))
    model.add(LSTM(512, kernel_regularizer=l2(0.0005)))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model

if __name__ == "__main__":
    # Télécharger les données du S&P 500
    data = fetch_data_from_yf(ticker='^GSPC', start='2005-01-01', end='2024-12-31')
    print("Données téléchargées avec succès :")
    print(data.head())

    # Ajouter des indicateurs techniques
    data = add_technical_indicators(data)

    # Paramètres
    look_back = 7

    # Préparer les données pour l'entraînement et le test
    X_train, Y_train, X_test, Y_test, scaler_X, scaler_y, dates_test, full_data, features = prepare_data(
        data, look_back
    )

    # Construire le modèle
    model = build_model(look_back, X_train.shape[2])

    # Définir les callbacks pour réduire le taux d'apprentissage et arrêter l'entraînement tôt
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraîner le modèle
    print("Entraînement du modèle...")
    history = model.fit(
        X_train, Y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, Y_test),
        callbacks=[reduce_lr, early_stopping]
    )

    # Prédictions sur les données de test
    predictions = model.predict(X_test)

    # Convertir les prévisions inversées à leur échelle d'origine
    inv_Y_test = scaler_y.inverse_transform(Y_test)
    inv_preds = scaler_y.inverse_transform(predictions)

    # Calculer les erreurs
    mae = mean_absolute_error(inv_Y_test, inv_preds)
    rmse = np.sqrt(mean_squared_error(inv_Y_test, inv_preds))
    print(f"Mean Absolute Error : {mae}")
    print(f"Root Mean Squared Error : {rmse}")

    # Vérifier la taille de dates_test et inv_preds
    print(f"Taille de dates_test : {len(dates_test)}")
    print(f"Taille de inv_preds : {len(inv_preds)}")

    # Limiter inv_preds à la taille de dates_test
    inv_preds_limited = inv_preds[:len(dates_test), 0]

    # Tracer la courbe du S&P 500 réel et la courbe prédite avec Plotly
    fig = go.Figure()

    # Ajouter la courbe réelle du S&P 500
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Réel S&P 500', line=dict(color='blue')))

    # Ajouter la courbe des prédictions
    fig.add_trace(go.Scatter(x=dates_test, y=inv_preds_limited, mode='lines', name='Prédictions', line=dict(color='orange')))

    # Mettre à jour les titres et étiquettes
    fig.update_layout(
        title='Prédiction et Réalité des Cours du S&P 500',
        xaxis_title='Date',
        yaxis_title='Prix de Clôture',
        legend_title="Légende",
        template="plotly_dark",
        hovermode="x unified"  # Afficher les informations au survol
    )

    # Afficher le graphique
    fig.show()
