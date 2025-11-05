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
    
    # Inicializar session_state si no existe
    if 'datos_procesados' not in st.session_state:
        st.session_state.datos_procesados = None
        st.session_state.df_original = None
        st.session_state.df_dias = None
        st.session_state.df_dias_var = None
        st.session_state.orden_estaciones = None
    
    # Botón para generar gráficos
    if st.button("🎨 Generar Gráficos"):
        try:
            # Cargar datos desde el archivo local
            df = pd.read_csv("joined_weather_data.csv")
        except Exception as e:
            st.error(f"Error al cargar el archivo csv: {e}")
            st.stop()
        
        # Convertir datetime_completo a formato datetime
        if 'datetime_completo' in df.columns:
            df['datetime_completo'] = pd.to_datetime(df['datetime_completo'])
        
        st.success(f"✅ Datos cargados: {len(df)} registros")
        
        # Mostrar muestra de datos
        with st.expander("👀 Ver muestra de datos"):
            st.dataframe(df.head(10))
        
        # ========== PREPROCESAMIENTO: ESTACIONES Y CONDICIONES ==========
        
        # Función para obtener estación
        def obtener_estacion(fecha):
            mes = fecha.month
            if mes in [12, 1, 2]:
                return 'Verano'
            elif mes in [3, 4, 5]:
                return 'Otoño'
            elif mes in [6, 7, 8]:
                return 'Invierno'
            else:
                return 'Primavera'
        
        # Crear columna de día (sin hora)
        df['dia'] = df['datetime_completo'].dt.date
        df['dia'] = pd.to_datetime(df['dia'])
        df['estacion'] = df['dia'].apply(obtener_estacion)
        
        # Detectar lluvia por hora
        lluvia_keywords = [
            'Rain', 'Drizzle', 'Showers', 'Thunderstorm',
            'Precipitation', 'Rain And Snow', 'Drizzle/Rain'
        ]
        df['lluvia_hora'] = df['conditions'].str.contains('|'.join(lluvia_keywords), case=False, na=False)
        
        # Agregación diaria para temperaturas (primera visualización)
        df_dias = (
            df.groupby(['dia', 'estacion'], as_index=False)
            .agg({
                'temp': ['max', 'min', 'mean', 'std'],
                'lluvia_hora': 'any'
            })
        )
        
        # Aplanar nombres de columnas
        df_dias.columns = ['dia', 'estacion', 'temp_max_dia', 'temp_min_dia', 'temp_avg_dia', 'temp_std_dia', 'lluvia_dia']
        
        # Crear condición_dia categórica
        df_dias['condicion_dia'] = df_dias['lluvia_dia'].map({False: 'Seco', True: 'Lluvioso'})
        
        # Orden de estaciones y condiciones
        orden_estaciones = ['Verano', 'Otoño', 'Invierno', 'Primavera']
        df_dias['estacion'] = pd.Categorical(df_dias['estacion'], categories=orden_estaciones, ordered=True)
        df_dias['condicion_dia'] = pd.Categorical(df_dias['condicion_dia'], categories=['Seco', 'Lluvioso'], ordered=True)
        
        # Agregación diaria para variabilidad (igual que en Colab)
        df_dias_var = (
            df.groupby(['dia', 'estacion'], as_index=False)
            .agg({
                'temp': 'std',
                'lluvia_hora': 'any'
            })
            .rename(columns={'temp': 'temp_std_dia', 'lluvia_hora': 'lluvia_dia'})
        )
        
        # Mapear a etiquetas legibles y tipo categoría
        df_dias_var['condicion_dia'] = pd.Categorical(
            df_dias_var['lluvia_dia'].map({False: 'Seco', True: 'Lluvioso'}),
            categories=['Seco', 'Lluvioso'],
            ordered=True
        )
        
        df_dias_var['estacion'] = pd.Categorical(df_dias_var['estacion'], categories=orden_estaciones, ordered=True)
        
        # Guardar en session_state
        st.session_state.datos_procesados = True
        st.session_state.df_original = df.copy()  # Guardar dataframe original para la nueva visualización
        st.session_state.df_dias = df_dias
        st.session_state.df_dias_var = df_dias_var
        st.session_state.orden_estaciones = orden_estaciones
    
    # Mostrar visualizaciones solo si los datos han sido procesados
    if st.session_state.datos_procesados and st.session_state.df_dias is not None:
        df_dias = st.session_state.df_dias
        orden_estaciones = st.session_state.orden_estaciones
        
        # ========== VISUALIZACIÓN 1: Temperatura Máxima por Estación ==========
        st.markdown("---")
        st.header("1️⃣ Comparación de Temperaturas según Condición Climática y Estación")
        
        st.markdown("""
        **Hipótesis:** Los días con lluvia tienen temperaturas máximas, mínimas y promedio menores que días secos.
        Análisis desagregado por estaciones del año.
        """)
        
        # Selector de tipo de temperatura (fuera del bloque del botón)
        tipo_temp = st.selectbox(
            'Seleccione el tipo de temperatura:',
            options=['Temperatura Máxima', 'Temperatura Mínima', 'Temperatura Promedio'],
            key='tipo_temp_selector'
        )
        
        # Mapear selección a columna y color
        temp_config = {
            'Temperatura Máxima': {'col': 'temp_max_dia', 'color': 'Reds', 'title': 'Temp. máxima'},
            'Temperatura Mínima': {'col': 'temp_min_dia', 'color': 'Blues', 'title': 'Temp. mínima'},
            'Temperatura Promedio': {'col': 'temp_avg_dia', 'color': 'Greens', 'title': 'Temp. promedio'}
        }
        
        config = temp_config[tipo_temp]
        
        # Crear gráficos individuales por estación (uno debajo del otro)
        for estacion in orden_estaciones:
            df_estacion = df_dias[df_dias['estacion'] == estacion]
            
            chart_temp = alt.Chart(df_estacion).mark_boxplot(size=60, opacity=0.8).encode(
                x=alt.X('condicion_dia:N', 
                        title='Condición del día',
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f'{config["col"]}:Q', 
                        title='Temperatura (°C)',
                        scale=alt.Scale(zero=False)),
                color=alt.Color('condicion_dia:N',
                                title='Condición',
                                scale=alt.Scale(
                                    domain=['Seco', 'Lluvioso'],
                                    range=['#E74C3C', '#3498DB']
                                )),
                tooltip=[
                    alt.Tooltip('condicion_dia:N', title='Condición'),
                    alt.Tooltip(f'mean({config["col"]}):Q', title='Media', format='.1f'),
                    alt.Tooltip(f'median({config["col"]}):Q', title='Mediana', format='.1f'),
                    alt.Tooltip('count():Q', title='N° días')
                ]
            ).properties(
                width=600,
                height=300,
                title=f'{config["title"]} - {estacion}'
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=13
            ).configure_legend(
                labelFontSize=12,
                titleFontSize=13
            ).configure_title(
                fontSize=15,
                anchor='start'
            )
            
            st.altair_chart(chart_temp, use_container_width=True)

        # Explicación de la hipótesis
        st.subheader("Explicación de la hipótesis 1")
        st.markdown("""
        Se considera esta hipótesis como verdadera, ya que se puede apreciar en general que tanto las medianas como los
        rangos intercuartílicos son menores en los días con lluvia que en los días sin lluvia.
                    
        Esta hipótesis fue muy informativa durante la exploración de los datos, ya que inicialmente no se consideraba importante
        la estación. Un primer acercamiento a esta hipótesis parecía mostrar que estábamos equivocados, ya que no se había separado
        por estaciones y los días lluviosos tenían temperaturas aparentemente mayores. Sin embargo, un anális del contexto meteorológico nos llevó
        a determinar que la mayor cantidad de días de lluvia se daban en verano, y en esa estación las temperaturas son en promedio
        más elevadas que el resto. Este detalle estaba sesgando nuestros gráficos.
                    
        Al tener esto en cuenta, se separó a los gráficos por estación, confirmando tanto la hipótesis como la razón por el
        sesgo inicial.
        """)
        
        # Estadísticas por estación y condición
        # st.subheader("📈 Estadísticas Descriptivas")
        
        #stats_temp = df_dias.groupby(['estacion', 'condicion_dia'])[config['col']].agg([
        #    ('Media', 'mean'),
        #    ('Mediana', 'median'),
        #    ('Desv.Std', 'std'),
        #    ('Mínima', 'min'),
        #    ('Máxima', 'max'),
        #    ('N° Días', 'count')
        #]).round(2)
        
        #st.dataframe(stats_temp, use_container_width=True)
        
        # ========== VISUALIZACIÓN 2: Temperatura Máxima Clear vs Cloudy ==========
        st.markdown("---")
        st.header("2️⃣ Comparación de Temperatura Máxima: Clear vs Cloudy")
        
        st.markdown("""
        **Hipótesis:** La temperatura máxima es significativamente mayor en días 'Clear' que en días 'Cloudy'.
        """)
        
        # Obtener dataframe original
        if 'df_original' in st.session_state and st.session_state.df_original is not None:
            df_original = st.session_state.df_original.copy()
            
            # Crear columna de día si no existe
            if 'dia' not in df_original.columns:
                df_original['dia'] = pd.to_datetime(df_original['datetime_completo'].dt.date)
            
            # Agregar por día y condición para obtener temperatura máxima diaria
            # Primero determinar qué día tiene qué condición (tomar la condición más frecuente del día)
            def obtener_condicion_dia(x):
                modes = x.mode()
                if len(modes) > 0:
                    return modes[0]
                else:
                    return x.iloc[0] if len(x) > 0 else None
            
            df_dia_condicion = df_original.groupby('dia')['conditions'].agg(obtener_condicion_dia).reset_index()
            df_dia_condicion.columns = ['dia', 'conditions']
            
            # Obtener temperatura máxima por día
            df_temp_max = df_original.groupby('dia', as_index=False).agg({
                'temp': 'max'
            }).rename(columns={'temp': 'temp_max_dia'})
            
            # Combinar con condiciones
            df_temp_max = df_temp_max.merge(df_dia_condicion, on='dia', how='left')
            
            # Filtrar datos para días Clear y Cloudy
            dias_clear = df_temp_max[df_temp_max['conditions'] == 'Clear'].copy()
            dias_cloudy = df_temp_max[df_temp_max['conditions'].str.contains('cloudy', case=False, na=False)].copy()
            
            if not dias_clear.empty and not dias_cloudy.empty:
                # Crear columna de condición para el gráfico
                dias_clear['condicion'] = 'Clear'
                dias_cloudy['condicion'] = 'Cloudy'
                
                # Combinar datos para el gráfico
                df_comparacion = pd.concat([dias_clear[['temp_max_dia', 'condicion']], 
                                           dias_cloudy[['temp_max_dia', 'condicion']]], 
                                          ignore_index=True)
                
                # Estadísticas descriptivas
                st.subheader("📈 Estadísticas Descriptivas - Temperatura Máxima")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**DÍAS CLEAR:**")
                    st.write(f"Promedio: {dias_clear['temp_max_dia'].mean():.2f}°C")
                    st.write(f"Mediana: {dias_clear['temp_max_dia'].median():.2f}°C")
                    st.write(f"Mínimo: {dias_clear['temp_max_dia'].min():.2f}°C")
                    st.write(f"Máximo: {dias_clear['temp_max_dia'].max():.2f}°C")
                    st.write(f"N° días: {len(dias_clear)}")
                
                with col2:
                    st.markdown("**DÍAS CLOUDY:**")
                    st.write(f"Promedio: {dias_cloudy['temp_max_dia'].mean():.2f}°C")
                    st.write(f"Mediana: {dias_cloudy['temp_max_dia'].median():.2f}°C")
                    st.write(f"Mínimo: {dias_cloudy['temp_max_dia'].min():.2f}°C")
                    st.write(f"Máximo: {dias_cloudy['temp_max_dia'].max():.2f}°C")
                    st.write(f"N° días: {len(dias_cloudy)}")
                
                diferencia = dias_clear['temp_max_dia'].mean() - dias_cloudy['temp_max_dia'].mean()
                st.markdown(f"**Diferencia de medias (Clear - Cloudy): {diferencia:.2f}°C**")
                
                # Gráfico de histogramas superpuestos
                chart_hist = alt.Chart(df_comparacion).mark_bar(opacity=0.7).encode(
                    x=alt.X('temp_max_dia:Q',
                            bin=alt.Bin(maxbins=15),
                            title='Temperatura Máxima (°C)'),
                    y=alt.Y('count():Q', title='Frecuencia'),
                    color=alt.Color('condicion:N',
                                    title='Condición',
                                    scale=alt.Scale(
                                        domain=['Clear', 'Cloudy'],
                                        range=['#FF8C00', '#808080']  # Naranja y gris
                                    )),
                    tooltip=[
                        alt.Tooltip('condicion:N', title='Condición'),
                        alt.Tooltip('temp_max_dia:Q', title='Temp. Máx (°C)', format='.1f'),
                        alt.Tooltip('count():Q', title='Frecuencia')
                    ]
                ).properties(
                    width=800,
                    height=400,
                    title='Distribución de Temperatura Máxima'
                ).configure_axis(
                    labelFontSize=12,
                    titleFontSize=13
                ).configure_legend(
                    labelFontSize=12,
                    titleFontSize=13
                ).configure_title(
                    fontSize=15,
                    anchor='start'
                )
                
                st.altair_chart(chart_hist, use_container_width=True)

                # Explicación de la hipótesis
                st.subheader("Explicación de la hipótesis 2")
                st.markdown("""
                En esta hipótesis, se buscaba comparar la temperatura máxima entre días Despejados y Nublados.
                Se planteó que la temperatura máxima sería significativamente mayor en días Despejados frente a los Nublados.
                
                Si bien un análisis de los datos demostró que la hipótesis no se cumple necesariamente,
                un análisis de las frecuencias demostró que las temperaturas máximas altas son mucho más frecuentes en días Despejados
                frente a días Nublados.
                """)
                
            else:
                st.warning("No se encontraron suficientes datos para días Clear o Cloudy.")