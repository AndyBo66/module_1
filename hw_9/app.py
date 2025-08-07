import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_model

st.title("Прогноз цены недвижимости")

# Загружаем модель и трансформер
model, transformer = load_model()
features = ['total_square', 'rooms', 'floor']

total_square = st.number_input('Общая площадь (м²)', min_value=10.0, max_value=500.0, value=50.0)
rooms = st.number_input('Количество комнат', min_value=1, max_value=10, value=3)
floor = st.number_input('Этаж', min_value=1, max_value=50, value=5)

# Формируем numpy массив
new_values = np.array([[total_square, rooms, floor]])

# Преобразуем в DataFrame с нужными именами колонок
new_data_df = pd.DataFrame(new_values, columns=features)
# Преобразуем через pipeline
new_data_transformed = transformer.transform(new_data_df)

st.image("imgs/flats.png", use_container_width=True)

if st.button('Предсказать цену'):
    prediction = model.predict(new_data_transformed)
    st.success(f'Предсказанная цена: {prediction[0]:,.0f} рублей')