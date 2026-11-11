# Прогнозирование стоимости недвижимости

Проект предназначен для прогнозирования стоимости недвижимости с использованием машинного обучения. Модель построена на основе `RandomForestRegressor`, а для получения прогнозов используется REST API на FastAPI.

## Структура проекта

```text
hw_10/
├── data/
│   ├── README.md
│   └── realty_data.csv
├── model/
│   ├── README.md
│   └── real_estate_model.joblib
├── main.py
├── train.py
├── requirements.txt
├── README.md
├── health.png
├── predict_get_1.png
├── predict_get_2.png
├── predict_post_1.png
└── predict_post_2.png
```

**Примечание:** файлы `realty_data.csv` и `real_estate_model.joblib` не включены в GitHub из-за большого размера. Они создаются или добавляются локально.

## 1. Установка

Рекомендуется Python 3.10 или новее.

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## 2. Подготовка данных

Поместите исходный файл `realty_data.csv` в папку `data/`.

Ожидаемый путь:

```text
data/realty_data.csv
```

## 3. Обучение модели

Из корневой папки проекта выполните:

```bash
python train.py
```

После обучения модель будет сохранена в:

```text
model/real_estate_model.joblib
```

В консоли будут выведены метрики качества модели:

- MAE — средняя абсолютная ошибка.
- R² — коэффициент детерминации.

## 4. Запуск API

Запустите сервер:

```bash
uvicorn main:app --reload
```

Документация Swagger доступна по адресу:

http://127.0.0.1:8000/docs

## 5. Проверка API endpoints

### GET /predict_get

Пример параметров запроса:

- `total_square`: 75
- `rooms`: 3
- `floor`: 5
- `lat`: 55.7558
- `lon`: 37.6173
- `city`: Москва
- `district`: Пресненский район

### POST /predict_post

Пример JSON-запроса:

```json
{
  "total_square": 75,
  "rooms": 3,
  "floor": 5,
  "lat": 55.7558,
  "lon": 37.6173,
  "city": "Москва",
  "district": "Пресненский район",
  "settlement": null
}
```

### GET /health

Проверка состояния API:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Если `model_loaded` равно `false`, сначала выполните обучение модели:

```bash
python train.py
```

После этого перезапустите API.