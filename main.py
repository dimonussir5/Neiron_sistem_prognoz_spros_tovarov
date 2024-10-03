from data_preprocessing import load_and_preprocess_data
from model import create_model, train_model
from evaluation import evaluate_model

def main():
    # Загрузка и предобработка данных
    file_path = 'sales_data.csv'
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
    
    # Создание модели
    model = create_model(input_dim=X_train.shape[1])
    
    # Обучение модели
    history = train_model(model, X_train, y_train)
    
    # Оценка модели
    mse, mae = evaluate_model(model, X_test, y_test)
    
    # Сохранение модели
    model.save('sales_demand_model.h5')

if __name__ == "__main__":
    main()