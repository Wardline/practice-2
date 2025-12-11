import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


#Пути к файлам
MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"

app = FastAPI(title="Real Estate Price API")


#Обучение и загрузка модели
def train_model():

    df = pd.read_csv(DATA_PATH)

    numeric_features = ["rooms", "total_square"]
    categorical_features = ["area"]  # строковый столбец, который мы OHE-кодируем

    #выкидываем строки с пропусками
    df = df[numeric_features + categorical_features + ["price"]].dropna()

    X = df[numeric_features + categorical_features]
    y = df["price"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("reg", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Файл {DATA_PATH} не найден."
        )

    if not os.path.exists(MODEL_PATH):
        print("model.pkl не найден - обучаем модель...")
        return train_model()

    print("Загружаем модель из model.pkl...")
    return joblib.load(MODEL_PATH)


model = load_model()  # глобальный объект модели


def predict_price(rooms: int, total_square: float, area: str) -> float:
    df = pd.DataFrame(
        {
            "rooms": [rooms],
            "total_square": [total_square],
            "area": [area],
        }
    )
    price = model.predict(df)[0]
    return float(price)


#схема для POST-запроса
class FlatFeatures(BaseModel):
    rooms: int
    total_square: float
    area: str

#Эндпоинты
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict_get")
def predict_get(
    rooms: int,
    total_square: float,
    area: str,
):
    #Пример:
    #/predict_get?rooms=2&total_square=45&area=Восточное%20Бутово%20м-н
    price = predict_price(rooms, total_square, area)
    return {"price": price}


@app.post("/predict_post")
def predict_post(features: FlatFeatures):
    #Получение предсказания через POST-запрос (JSON)
    price = predict_price(
        rooms=features.rooms,
        total_square=features.total_square,
        area=features.area,
    )
    return {"price": price}
