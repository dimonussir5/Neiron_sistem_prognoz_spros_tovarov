import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test):
    # Прогнозирование на тестовой выборке
    y_pred = model.predict(X_test)
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Фактические продажи')
    plt.plot(y_pred, label='Прогнозируемые продажи')
    plt.legend()
    plt.show()
    
    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    
    return mse, mae