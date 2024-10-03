import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(num_rows=10000):
    # Начальная дата
    start_date = datetime(2023, 1, 1)
    
    # Генерация данных
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]
    sales = np.random.randint(50, 200, num_rows)
    prices = np.random.randint(40, 70, num_rows)
    advertising_costs = np.random.randint(800, 2000, num_rows)
    
    # Создание DataFrame
    data = {
        'date': dates,
        'sales': sales,
        'price': prices,
        'advertising_cost': advertising_costs
    }
    df = pd.DataFrame(data)
    
    # Сохранение в CSV файл
    df.to_csv('sales_data.csv', index=False)

# Генерация данных и сохранение в файл
generate_sales_data()