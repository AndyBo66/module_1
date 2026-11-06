from pathlib import Path
import json
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "real_estate_model.pkl"
INFO_PATH = BASE_DIR / "model" / "model_info.json"

st.set_page_config(page_title="Оценка недвижимости", page_icon="🏠", layout="centered")
st.title("🏠 Прогнозирование стоимости недвижимости")
st.write("Введите характеристики объекта и нажмите кнопку, чтобы получить оценку стоимости.")

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        info = json.loads(
            INFO_PATH.read_text(encoding="utf-8")
        )

        return model, info

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, info = load_model()

if model is None:
    st.error("Обученная модель не найдена. Сначала выполните команду: python train.py")
    st.stop()

defaults = info["numeric_defaults"]
options = info["category_options"]

st.subheader("Характеристики объекта")
col1, col2 = st.columns(2)

with col1:
    total_square = st.number_input(
        "Общая площадь, м²", min_value=1.0, max_value=1000.0,
        value=float(max(1.0, defaults.get("total_square", 60.0))), step=1.0
    )
    rooms = st.number_input(
        "Количество комнат", min_value=0.0, max_value=20.0,
        value=float(max(0.0, defaults.get("rooms", 2.0))), step=1.0
    )
    floor = st.number_input(
        "Этаж", min_value=0.0, max_value=200.0,
        value=float(max(0.0, defaults.get("floor", 5.0))), step=1.0
    )
    lat = st.number_input(
        "Широта (lat)", min_value=-90.0, max_value=90.0,
        value=float(defaults.get("lat", 55.75)), format="%.6f"
    )
    lon = st.number_input(
        "Долгота (lon)", min_value=-180.0, max_value=180.0,
        value=float(defaults.get("lon", 37.62)), format="%.6f"
    )

with col2:
    def category_input(label, key, fallback="Не указано"):
        values = options.get(key, [])
        choices = ["Не указано"] + [v for v in values if v != "Не указано"]
        if choices and len(choices) <= 100:
            return st.selectbox(label, choices, index=0)
        return st.text_input(label, value=fallback, key=f"input_{key}")

    city = category_input("Город", "city", "Москва")
    district = category_input("Район", "district")
    settlement = st.text_input("Населённый пункт", value="Не указано")
    object_type = category_input("Тип объекта", "object_type", "Квартира")
    source = category_input("Источник объявления", "source", "Не указано")

st.caption("Примечание: координаты и местоположение влияют на оценку. Указывайте их в формате широты и долготы.")

if st.button("Рассчитать стоимость", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "city": city,
        "district": district,
        "settlement": settlement,
        "object_type": object_type,
        "source": source
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        st.success("Прогноз рассчитан")
        st.metric("Предполагаемая стоимость", f"{prediction:,.0f} ₽".replace(",", " "))
        st.info("Это оценка модели, а не гарантированная рыночная цена.")
    except Exception as error:
        st.error(f"Не удалось выполнить прогноз: {error}")

with st.expander("О модели"):
    st.write(f"Алгоритм: RandomForestRegressor")
    st.write(f"Объектов для обучения: {info.get('training_rows', '—'):,}".replace(",", " "))
    st.write(f"MAE на тестовой выборке: {info.get('mae', float('nan')):,.0f} руб.".replace(",", " "))
    st.write(f"R² на тестовой выборке: {info.get('r2', float('nan')):.4f}")
