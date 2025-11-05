# main.py
# 🌤️ App de ejemplo: consulta del clima con Streamlit
import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Weather Data App", page_icon="🌦️", layout="centered")

st.title("🌤️ Aplicación del clima")
st.write("Consulta el clima actual de cualquier ciudad usando datos de OpenWeatherMap.")

# Entrada de ciudad
city = st.text_input("Ingrese una ciudad:")

# Botón para consultar
if st.button("Consultar clima"):
    if not city:
        st.warning("Por favor, ingrese una ciudad.")
    else:
        # API pública (ejemplo sin clave, limitado)
        url = f"https://wttr.in/{city}?format=j1"
        try:
            response = requests.get(url)
            data = response.json()

            # Obtener datos
            current = data["current_condition"][0]
            temp = current["temp_C"]
            weather_desc = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            feels_like = current["FeelsLikeC"]

            # Mostrar resultados
            st.subheader(f"Clima en {city.capitalize()}")
            st.metric("Temperatura", f"{temp} °C")
            st.write(f"**Sensación térmica:** {feels_like} °C")
            st.write(f"**Humedad:** {humidity}%")
            st.write(f"**Condición:** {weather_desc}")

        except Exception as e:
            st.error("No se pudo obtener la información del clima. Verifique la conexión o el nombre de la ciudad.")
            st.text(e)

st.markdown("---")
st.caption("Hecho con ❤️ en Streamlit")
