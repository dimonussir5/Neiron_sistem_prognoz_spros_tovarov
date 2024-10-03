import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)
    
    # Предобработка данных
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Разделение данных на признаки (X) и целевую переменную (y)
    X = data[['price', 'advertising_cost']]
    y = data['sales']
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler