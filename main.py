import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción del Clima", page_icon="🌦️", layout="wide")
st.title("🌤️ Predicción del Clima con Modelo de ML")

# Crear tabs
tab1, tab2 = st.tabs(["🔮 Predicción del Clima", "📊 Análisis de Datos"])

# ==================== TAB 1: PREDICCIÓN ====================
with tab1:
    st.header("Predicción del Clima")
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

# ==================== TAB 2: VISUALIZACIONES ====================
with tab2:
    st.header("📊 Análisis de Datos Climáticos")
    
    st.write("Genera gráficos interactivos usando los datos históricos del clima en Mendoza.")
    
    # Botón para generar gráficos
    if st.button("🎨 Generar Gráficos"):
        try:
            # Cargar datos desde el archivo local
            df = pd.read_csv("joined_weather_data.csv")
        except e:
            st.write("Error al cargar el archivo csv: " + e)
        
        # Convertir datetime_completo a formato datetime
        if 'datetime_completo' in df.columns:
            df['datetime_completo'] = pd.to_datetime(df['datetime_completo'])
        
        st.success(f"✅ Datos cargados: {len(df)} registros")
        
        # Mostrar muestra de datos
        with st.expander("👀 Ver muestra de datos"):
            st.dataframe(df.head(10))
        
        # Crear categoría de condición climática
        def categorizar_clima(conditions):
            if pd.isna(conditions):
                return 'Desconocido'
            conditions_lower = str(conditions).lower()
            if 'rain' in conditions_lower or 'lluvia' in conditions_lower:
                return 'Lluvia'
            elif 'storm' in conditions_lower or 'tormenta' in conditions_lower:
                return 'Tormenta'
            elif 'clear' in conditions_lower or 'despejado' in conditions_lower:
                return 'Despejado'
            elif 'cloud' in conditions_lower or 'nublado' in conditions_lower:
                return 'Nublado'
            else:
                return 'Otro'

        if 'conditions' in df.columns:
            df['categoria_clima'] = df['conditions'].apply(categorizar_clima)
        
        # ========== VISUALIZACIÓN 1: Comparación de Temperaturas ==========
        st.markdown("---")
        st.header("1️⃣ Comparación de Temperaturas según Condición Climática")
        
        st.markdown("""
        **Hipótesis:** Los días con lluvia tienen temperaturas máximas, mínimas y promedio menores que días secos.
        """)
        
        if 'categoria_clima' in df.columns and 'temp' in df.columns:
            # Selector interactivo de categorías
            categorias_disponibles = df['categoria_clima'].unique().tolist()
            categorias_seleccionadas = st.multiselect(
                'Seleccione condición(es) climática(s):',
                options=categorias_disponibles,
                default=categorias_disponibles,
                key='categorias_viz1'
            )
            
            df_filtered = df[df['categoria_clima'].isin(categorias_seleccionadas)]
            
            # Crear gráfico con estadísticas
            base = alt.Chart(df_filtered).encode(
                x=alt.X('categoria_clima:N', 
                        title='Condición Climática',
                        axis=alt.Axis(labelAngle=-15)),
                color=alt.Color('categoria_clima:N',
                                title='Condición',
                                scale=alt.Scale(scheme='tableau10'))
            )
            
            # Boxplot para temperatura
            boxplot = base.mark_boxplot(size=40, opacity=0.7).encode(
                y=alt.Y('temp:Q', 
                        title='Temperatura (°C)',
                        scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip('categoria_clima:N', title='Condición'),
                    alt.Tooltip('mean(temp):Q', title='Temp. Media', format='.1f'),
                    alt.Tooltip('median(temp):Q', title='Mediana', format='.1f')
                ]
            )
            
            # Puntos individuales
            points = base.mark_circle(size=30, opacity=0.3).encode(
                y=alt.Y('temp:Q'),
                xOffset='jitter:Q',
                tooltip=[
                    alt.Tooltip('datetime_completo:T', title='Fecha', format='%Y-%m-%d'),
                    alt.Tooltip('categoria_clima:N', title='Condición'),
                    alt.Tooltip('temp:Q', title='Temperatura', format='.1f'),
                    alt.Tooltip('feelslike:Q', title='Sensación', format='.1f'),
                    alt.Tooltip('precipprob:Q', title='Prob. Precip.', format='.0f')
                ] if 'feelslike' in df.columns and 'precipprob' in df.columns else [
                    alt.Tooltip('datetime_completo:T', title='Fecha', format='%Y-%m-%d'),
                    alt.Tooltip('categoria_clima:N', title='Condición'),
                    alt.Tooltip('temp:Q', title='Temperatura', format='.1f')
                ]
            ).transform_calculate(
                jitter='sqrt(-2*log(random()))*cos(2*PI*random())*8'
            )
            
            chart1 = (boxplot + points).properties(
                width=700,
                height=400,
                title='Distribución de Temperatura según Condición Climática'
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            ).configure_legend(
                labelFontSize=12,
                titleFontSize=13
            ).configure_title(
                fontSize=16,
                anchor='start'
            )
            
            st.altair_chart(chart1, use_container_width=True)
            
            # Mostrar estadísticas descriptivas
            st.subheader("📈 Estadísticas de Temperatura por Condición")
            stats_temp = df.groupby('categoria_clima')['temp'].agg([
                ('Temperatura Media', 'mean'),
                ('Desv. Estándar', 'std'),
                ('Mínima', 'min'),
                ('Máxima', 'max'),
                ('N° Días', 'count')
            ]).round(2)
            st.dataframe(stats_temp, use_container_width=True)
        
        # ========== VISUALIZACIÓN 2: Variabilidad de Temperatura ==========
        st.markdown("---")
        st.header("2️⃣ Variabilidad de Temperatura según Condición Climática")
        
        st.markdown("""
        **Hipótesis:** Los días con lluvia o tormenta tienen menor variabilidad de temperatura que días despejados.
        """)
        
        if 'categoria_clima' in df.columns and 'temp' in df.columns and 'feelslike' in df.columns:
            # Calcular diferencia entre sensación térmica y temperatura real
            df['diferencia_sensacion'] = abs(df['temp'] - df['feelslike'])
            
            # Gráfico de barras con variabilidad
            base_var = alt.Chart(df).encode(
                x=alt.X('categoria_clima:N', 
                        title='Condición Climática',
                        axis=alt.Axis(labelAngle=-15)),
                color=alt.Color('categoria_clima:N',
                                title='Condición',
                                scale=alt.Scale(scheme='tableau10'))
            )
            
            # Barras de variabilidad promedio
            bars = base_var.mark_bar(opacity=0.7, size=50).encode(
                y=alt.Y('mean(diferencia_sensacion):Q',
                        title='Diferencia Promedio Temp - Sensación (°C)',
                        scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip('categoria_clima:N', title='Condición'),
                    alt.Tooltip('mean(diferencia_sensacion):Q', title='Diferencia Promedio', format='.2f'),
                    alt.Tooltip('count():Q', title='N° de días')
                ]
            )
            
            # Error bars
            error_bars = base_var.mark_errorbar(extent='stdev', ticks=True).encode(
                y=alt.Y('diferencia_sensacion:Q')
            )
            
            # Selección interactiva
            brush = alt.selection_interval(encodings=['x'])
            
            chart2_top = (bars + error_bars).encode(
                opacity=alt.condition(brush, alt.value(1), alt.value(0.3))
            ).add_params(brush).properties(
                width=700,
                height=350,
                title='Variabilidad de Sensación Térmica por Condición Climática'
            )
            
            # Gráfico de dispersión temporal detallado
            scatter_tooltip = [
                alt.Tooltip('datetime_completo:T', title='Fecha', format='%Y-%m-%d'),
                alt.Tooltip('categoria_clima:N', title='Condición'),
                alt.Tooltip('temp:Q', title='Temperatura', format='.1f'),
                alt.Tooltip('feelslike:Q', title='Sensación', format='.1f'),
                alt.Tooltip('diferencia_sensacion:Q', title='Diferencia', format='.1f')
            ]
            
            if 'humidity' in df.columns:
                scatter_tooltip.append(alt.Tooltip('humidity:Q', title='Humedad', format='.0f'))
            if 'windspeed' in df.columns:
                scatter_tooltip.append(alt.Tooltip('windspeed:Q', title='Viento', format='.1f'))
            
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('datetime_completo:T', title='Fecha'),
                y=alt.Y('diferencia_sensacion:Q', title='Diferencia (°C)'),
                color=alt.Color('categoria_clima:N', scale=alt.Scale(scheme='tableau10')),
                tooltip=scatter_tooltip
            ).transform_filter(
                brush
            ).properties(
                width=700,
                height=200,
                title='Detalle Temporal de Días Seleccionados'
            )
            
            chart2 = alt.vconcat(chart2_top, scatter).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            ).configure_legend(
                labelFontSize=12,
                titleFontSize=13
            ).configure_title(
                fontSize=16,
                anchor='start'
            )
            
            st.altair_chart(chart2, use_container_width=True)
            
            # Estadísticas de variabilidad
            st.subheader("📊 Análisis de Variabilidad")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Diferencia Temp - Sensación por Condición**")
                stats_var = df.groupby('categoria_clima')['diferencia_sensacion'].agg([
                    ('Promedio', 'mean'),
                    ('Desv. Estándar', 'std'),
                    ('Mínima', 'min'),
                    ('Máxima', 'max')
                ]).round(2)
                st.dataframe(stats_var, use_container_width=True)
            
            with col2:
                if 'humidity' in df.columns and 'windspeed' in df.columns:
                    st.markdown("**Humedad y Viento por Condición**")
                    stats_factores = df.groupby('categoria_clima')[['humidity', 'windspeed']].agg([
                        ('Media', 'mean'),
                        ('Desv.Std', 'std')
                    ]).round(2)
                    st.dataframe(stats_factores, use_container_width=True)
        
        # ========== VISUALIZACIÓN 3: Matriz de Correlación ==========
        st.markdown("---")
        st.header("3️⃣ Correlación entre Variables Climáticas")
        
        # Seleccionar variables disponibles
        vars_posibles = ['temp', 'feelslike', 'humidity', 'precipprob', 'windspeed', 'pressure', 'cloudcover']
        vars_interes = [var for var in vars_posibles if var in df.columns]
        
        if len(vars_interes) >= 2:
            df_corr = df[vars_interes].corr()
            
            # Convertir a formato largo para Altair
            df_corr_long = df_corr.reset_index().melt(id_vars='index')
            df_corr_long.columns = ['Variable 1', 'Variable 2', 'Correlación']
            
            heatmap = alt.Chart(df_corr_long).mark_rect().encode(
                x=alt.X('Variable 1:N', title=None),
                y=alt.Y('Variable 2:N', title=None),
                color=alt.Color('Correlación:Q', 
                                scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                                title='Correlación'),
                tooltip=[
                    alt.Tooltip('Variable 1:N'),
                    alt.Tooltip('Variable 2:N'),
                    alt.Tooltip('Correlación:Q', format='.3f')
                ]
            ).properties(
                width=500,
                height=500,
                title='Matriz de Correlación entre Variables Climáticas'
            )
            
            # Añadir valores de texto
            text = heatmap.mark_text(baseline='middle').encode(
                text=alt.Text('Correlación:Q', format='.2f'),
                color=alt.condition(
                    alt.datum.Correlación > 0.5,
                    alt.value('white'),
                    alt.value('black')
                )
            )
            
            chart3 = (heatmap + text).configure_axis(
                labelFontSize=11,
                labelAngle=-45
            ).configure_legend(
                labelFontSize=12,
                titleFontSize=13
            ).configure_title(
                fontSize=16,
                anchor='start'
            )
            
            st.altair_chart(chart3, use_container_width=True)
            
            st.markdown("""
            **Interpretación:** Esta matriz muestra las correlaciones entre diferentes variables climáticas,
            lo que puede ayudar a entender las relaciones subyacentes entre temperatura, humedad, precipitación y otras variables.
            """)
        else:
            st.warning("No hay suficientes variables numéricas para crear la matriz de correlación.")