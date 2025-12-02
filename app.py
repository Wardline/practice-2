import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Прогноз стоимости недвижимости")

MODEL_PATH = "model.pkl"
DATA_PATH = "data.csv"


#Обучение модели с OHE
def train_model():
    df = pd.read_csv(DATA_PATH)

    numeric_features = ["rooms", "total_square"]
    categorical_features = ["area"]

    #выбрасываем строки с NaN
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


def read_model():
    return joblib.load(MODEL_PATH)


#Если model.pkl нет - обучаем
if not os.path.exists(MODEL_PATH):
    if not os.path.exists(DATA_PATH):
        st.error("data.csv не найден!")
        st.stop()

    st.write("Обучаем модель (One-Hot Encoding)...")
    model = train_model()
else:
    model = read_model()


#Интерфейс Streamlit
st.title("Прогноз стоимости недвижимости")

#Ввод данных
rooms = st.sidebar.number_input("Количество комнат", min_value=1, max_value=10, value=2)
total_square = st.sidebar.number_input(
    "Площадь (м²)", min_value=10.0, max_value=300.0, value=50.0
)

#Подтягиваем уникальные районы (area) из CSV
df_full = pd.read_csv(DATA_PATH)
areas = sorted(df_full["area"].dropna().unique())

area = st.sidebar.selectbox("Район", areas)

st.write("---")

#Предсказание
if st.button("Рассчитать стоимость"):
    X_input = pd.DataFrame(
        {
            "rooms": [rooms],
            "total_square": [total_square],
            "area": [area],
        }
    )

    price = model.predict(X_input)[0]
    st.success(f"Оценочная стоимость: {price:,.0f} у.е.")
