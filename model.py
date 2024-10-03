import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_dim):
    # Создание модели
    model = Sequential()
    
    # Добавление слоев
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Линейная активация для регрессии
    
    # Компиляция модели
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train):
    # Обучение модели
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return history