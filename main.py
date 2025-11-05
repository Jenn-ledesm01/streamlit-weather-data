import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción del Clima", page_icon="🌦️", layout="centered")
st.title("🌤️ Predicción del Clima con Modelo de ML")

st.write("Ingrese su API Key y una fecha para obtener la predicción del clima en Mendoza, Argentina.")

# Entradas del usuario
api_key = st.text_input("🔑 Ingrese su API Key de Visual Crossing:")
fecha_actual = st.date_input("📅 Seleccione la fecha (YYYY-MM-DD):", datetime.today().date())

# Botón
if st.button("Predecir clima"):
    if not api_key:
        st.warning("Por favor ingrese su API key.")
    else:
        try:
            # Fechas
            fecha_actual_str = fecha_actual.strftime("%Y-%m-%d")
            fecha_ayer = (fecha_actual - timedelta(days=1)).strftime("%Y-%m-%d")

            # Ciudad fija
            location = "Mendoza,Argentina"
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{fecha_ayer}/{fecha_actual}"
            params = {
                "unitGroup": "metric",
                "include": "days",
                "contentType": "json",
                "key": api_key,
            }

            # Petición a la API
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()["days"]

            if len(data) < 2:
                st.error("No se obtuvieron datos suficientes (se necesitan 2 días).")
            else:
                # Día de ayer y hoy
                ayer, hoy = data[0], data[1]

                # Construir la fila con los valores requeridos
                features = {
                    "temp_mean": hoy["temp"],
                    "feelslike_mean": hoy["feelslike"],
                    "humidity_mean": hoy["humidity"],
                    "dew_mean": hoy["dew"],
                    "pressure_mean": hoy["pressure"],
                    "windspeed_mean": hoy["windspeed"],
                    "windgust_mean": hoy["windgust"],
                    "winddir_mean": hoy["winddir"],
                    "visibility_mean": hoy["visibility"],
                    "solarradiation_mean": hoy["solarradiation"],
                    "uvindex_mean": hoy["uvindex"],
                    "cloudcover_mean": hoy["cloudcover"],
                    "precip_sum": hoy["precip"],
                    "snow_sum": hoy["snow"],
                    "temp_range": hoy["tempmax"] - hoy["tempmin"],
                    "dew_point_diff": hoy["temp"] - hoy["dew"],

                    # Features cíclicas
                    "month_sin": np.sin(2 * np.pi * fecha_actual.month / 12),
                    "month_cos": np.cos(2 * np.pi * fecha_actual.month / 12),
                    "dayofyear_sin": np.sin(2 * np.pi * fecha_actual.timetuple().tm_yday / 365),
                    "dayofyear_cos": np.cos(2 * np.pi * fecha_actual.timetuple().tm_yday / 365),

                    # Lluvia ayer
                    "rain_yesterday": 1 if ayer["precip"] > 0 else 0,
                }

                X = pd.DataFrame([features])

                # Cargar el modelo
                model = joblib.load("model_output/gradient_boosting_weather_model.pkl")

                # Predicción (nombre exacto de la clase)
                pred = model.predict(X)[0]

                # Mostrar resultado
                st.subheader("🌦️ Resultado de la predicción:")
                if pred.lower() == "rain":
                    st.success("🌧️ Predicción: **Rain**")
                elif pred.lower() == "cloudy":
                    st.info("☁️ Predicción: **Cloudy**")
                elif pred.lower() == "clear":
                    st.warning("☀️ Predicción: **Clear**")
                else:
                    st.write(f"Predicción desconocida: {pred}")

                # Mostrar datos usados
                with st.expander("📊 Ver datos usados para la predicción"):
                    st.write(X)

        except Exception as e:
            st.error(f"Error al obtener datos o predecir: {e}")
