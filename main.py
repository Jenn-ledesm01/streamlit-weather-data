import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime, timedelta

API_KEYS = [
    "N9FENAZ4MC65WBZ6J6AWGULZ3",
    "54G4EHM72LT7762EHUQMKERYE",
    "5YXQ8PZG4HJQTG4WLQ4CYZBLJ",
    "LZCNRDCYVBUKWK79K3ZD3YVN9",
    "C97H3YUSQBF833J35FNMWHTLZ"
]

# Función para probar API keys
def obtener_datos_clima(location, fecha_ayer, fecha_actual):
    """Intenta obtener datos usando las API keys disponibles"""
    for idx, api_key in enumerate(API_KEYS):
        try:
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{fecha_ayer}/{fecha_actual}"
            params = {
                "unitGroup": "metric",
                "include": "days",
                "contentType": "json",
                "key": api_key,
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()["days"]
            
            # Si llegamos aquí, la API key funcionó
            return data, api_key, idx + 1
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too many requests
                st.warning(f"API Key {idx + 1} sin créditos. Probando siguiente...")
                continue
            else:
                raise e
        except Exception as e:
            if idx == len(API_KEYS) - 1:  # Última key
                raise e
            continue
    
    raise Exception("Todas las API keys agotaron sus créditos")

# Configuración de la página
st.set_page_config(page_title="Predicción del clima", page_icon="🌦️", layout="wide")
st.title("🌤️ Predicción del clima con modelo de Machine Learning")

# Crear tabs (agregando tab de inicio)
tab0, tab1, tab2 = st.tabs(["🏠 Inicio", "🔮 Predicción del clima", "📊 Exploración de datos"])

# ==================== TAB 0: INICIO ====================
with tab0:
    st.header("¡Bienvenido a la aplicación de predicción del clima! 👋")
    
    st.markdown("""
    Esta aplicación utiliza **Machine Learning** para predecir las condiciones climáticas en **Mendoza, Argentina** 
    y proporciona herramientas de análisis de datos históricos.
    """)
    
    st.markdown("---")
    
    # Sección: ¿Qué puedes hacer?
    st.subheader("🎯 ¿Qué puedes hacer en esta aplicación?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔮 Predicción del clima
        - **Predice** las condiciones climáticas para una fecha específica
        - Utiliza un modelo de **Gradient Boosting** entrenado con datos históricos
        - Obtén probabilidades para diferentes condiciones: **Clear**, **Cloudy**, **Rain**
        - Visualiza la distribución de probabilidades en un gráfico interactivo
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Exploración de datos
        - **Explora** patrones climáticos históricos de Mendoza
        - Compara temperaturas según condiciones climáticas y estaciones del año
        - Analiza la diferencia entre días **despejados** y **nublados**
        - Visualizaciones interactivas con **gráficos de caja** e **histogramas**
        """)
    
    st.markdown("---")
    
    # Sección: Cómo usar la app
    st.subheader("📖 Cómo usar esta aplicación")
    
    with st.expander("🔮 Pestaña: Predicción del clima"):
        st.markdown("""
        1. **Selecciona una fecha**: Elige la fecha para la cual deseas la predicción
        4. **Haz clic en "Predecir clima"**: El modelo procesará los datos y mostrará:
           - La condición climática más probable
           - Un gráfico de torta con las probabilidades de cada condición
           - Los datos utilizados para la predicción
        """)
    
    with st.expander("📊 Pestaña: Exploración de datos"):
        st.markdown("""
        1. **Explora las visualizaciones**:
           - **Visualización 1**: Compara temperaturas (máximas, mínimas o promedio) entre días secos y lluviosos, separadas por estación
           - **Visualización 2**: Analiza la diferencia de temperaturas máximas entre días despejados (Clear) y nublados (Cloudy)
        3. **Lee las explicaciones**: Cada visualización incluye el contexto y las conclusiones de las hipótesis planteadas
        """)
    
    st.markdown("---")
    
    # Sección: Sobre el modelo
    st.subheader("🤖 Sobre el Modelo de Machine Learning")
    
    st.markdown("""
    El modelo utilizado es un **Gradient Boosting Classifier** entrenado con datos climáticos históricos de Mendoza.
    
    **Características del modelo:**
    - 🎯 **Variables de entrada**: Temperatura, humedad, presión, viento, radiación solar, cobertura de nubes, y más
    - 🔄 **Features cíclicas**: Representación sinusoidal del mes y día del año para capturar patrones estacionales
    - 📅 **Contexto temporal**: Incluye información del día anterior (como lluvia previa)
    - 🎲 **Salida**: Probabilidades para tres condiciones climáticas principales: Clear, Cloudy y Rain
    
    **Ventajas del Gradient Boosting:**
    - Alta precisión en problemas de clasificación
    - Capacidad para capturar relaciones no lineales
    - Resistencia al overfitting mediante técnicas de regularización
    """)
    
    st.markdown("---")
    
    # Sección: Datos
    st.subheader("📁 Sobre los Datos")
    
    st.markdown("""
    - **Fuente**: Visual Crossing Weather API
    - **Ubicación**: Mendoza, Argentina
    - **Período**: Datos históricos utilizados para entrenamiento y análisis
    - **Variables**: Temperatura, humedad, precipitación, viento, presión, radiación solar, índice UV, cobertura de nubes, visibilidad y más
    - **Frecuencia**: Datos por hora agregados a nivel diario para análisis
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Desarrollado usando Streamlit, Scikit-learn y Altair</p>
        <p><small>Los resultados son predicciones basadas en datos históricos y no deben usarse como única fuente para decisiones críticas</small></p>
    </div>
    """, unsafe_allow_html=True)

# ==================== TAB 1: PREDICCIÓN ====================
with tab1:
    st.header("Predicción del clima")

    # Entradas del usuario
    fecha_actual = st.date_input("📅 Seleccione la fecha (YYYY-MM-DD):", datetime.today().date())

    # Ejecutar predicción automáticamente al seleccionar la fecha
    try:
        # Fechas
        fecha_actual_str = fecha_actual.strftime("%Y-%m-%d")
        fecha_ayer = (fecha_actual - timedelta(days=1)).strftime("%Y-%m-%d")

        # Ciudad fija
        location = "Mendoza,Argentina"
        
        # Obtener datos usando las API keys (con rotación automática)
        with st.spinner("Obteniendo datos del clima..."):
            data, api_key_usada, numero_key = obtener_datos_clima(location, fecha_ayer, fecha_actual_str)

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

            # ================= PREDICCIÓN Y PROBABILIDADES =================
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            clases = model.classes_

            # Mostrar resultado principal destacado
            st.subheader("🌦️ Resultado de la predicción:")

            if pred.lower() == "rain":
                st.markdown(
                    "<div style='background-color:#D0E8FF; padding:15px; border-radius:10px; text-align:center;'>"
                    "<h2 style='color:#007BFF;'>🌧️ Predicción más probable: <b>Rain</b></h2>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif pred.lower() == "cloudy":
                st.markdown(
                    "<div style='background-color:#E8E8E8; padding:15px; border-radius:10px; text-align:center;'>"
                    "<h2 style='color:#555;'>☁️ Predicción más probable: <b>Cloudy</b></h2>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif pred.lower() == "clear":
                st.markdown(
                    "<div style='background-color:#FFF4C2; padding:15px; border-radius:10px; text-align:center;'>"
                    "<h2 style='color:#E0A800;'>☀️ Predicción más probable: <b>Clear</b></h2>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background-color:#F8F9FA; padding:15px; border-radius:10px; text-align:center;'>"
                    f"<h2>🔍 Predicción más probable: <b>{pred}</b></h2>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ================= GRÁFICO DE TORTA INTERACTIVO =================
            st.markdown("### 📊 Distribución de probabilidades")

            # Crear DataFrame con las probabilidades
            df_probs = pd.DataFrame({
                "Condición": clases,
                "Probabilidad": np.round(probs * 100, 2)
            })

            # Crear gráfico de torta (pie chart) con Altair
            chart = (
                alt.Chart(df_probs)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("Probabilidad:Q", title="Probabilidad (%)"),
                    color=alt.Color("Condición:N", legend=alt.Legend(title="Condición climática")),
                    tooltip=[
                        alt.Tooltip("Condición:N", title="Condición"),
                        alt.Tooltip("Probabilidad:Q", title="Probabilidad (%)")
                    ]
                )
                .properties(width=400, height=400)
                .interactive()  # permite zoom y hover
            )

            # Mostrar el gráfico
            st.altair_chart(chart, use_container_width=True)


            # Mostrar datos usados
            with st.expander("📊 Ver datos usados para la predicción"):
                st.write(X)

    except Exception as e:
        st.error(f"Error al obtener datos o predecir: {e}")

# ==================== TAB 2: VISUALIZACIONES ====================
with tab2:    
    # Inicializar session_state si no existe
    if 'datos_procesados' not in st.session_state:
        st.session_state.datos_procesados = None
        st.session_state.df_original = None
        st.session_state.df_dias = None
        st.session_state.orden_estaciones = None
    
    # Procesar datos automáticamente si no están en session_state
    if not st.session_state.datos_procesados or st.session_state.df_dias is None:
        try:
            with st.spinner("Cargando y procesando datos..."):
                # Cargar datos desde el archivo local
                df = pd.read_csv("joined_weather_data.csv")
                
                # Convertir datetime_completo a formato datetime
                if 'datetime_completo' in df.columns:
                    df['datetime_completo'] = pd.to_datetime(df['datetime_completo'])
                
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
                df['mes'] = df['dia'].dt.month
                df['mes_nombre'] = df['dia'].dt.strftime('%B')
                df['estacion'] = df['dia'].apply(obtener_estacion)
                
                # Detectar lluvia por hora
                lluvia_keywords = [
                    'Rain', 'Drizzle', 'Showers', 'Thunderstorm',
                    'Precipitation', 'Rain And Snow', 'Drizzle/Rain'
                ]
                df['lluvia_hora'] = df['conditions'].str.contains('|'.join(lluvia_keywords), case=False, na=False)
                
                # Agregación diaria para temperaturas
                df_dias = (
                    df.groupby(['dia', 'estacion', 'mes', 'mes_nombre'], as_index=False)
                    .agg({
                        'temp': ['max', 'min', 'mean'],
                        'feelslike': 'mean',
                        'humidity': 'mean',
                        'lluvia_hora': 'any',
                        'conditions': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
                    })
                )
                
                # Aplanar nombres de columnas
                df_dias.columns = ['dia', 'estacion', 'mes', 'mes_nombre', 'temp_max_dia', 'temp_min_dia', 
                                   'temp_avg_dia', 'feelslike_avg', 'humidity_avg', 'lluvia_dia', 'conditions']
                
                # Crear condición_dia categórica
                df_dias['condicion_dia'] = df_dias['lluvia_dia'].map({False: 'Seco', True: 'Lluvioso'})
                
                # Orden de estaciones
                orden_estaciones = ['Verano', 'Otoño', 'Invierno', 'Primavera']
                df_dias['estacion'] = pd.Categorical(df_dias['estacion'], categories=orden_estaciones, ordered=True)
                
                # Guardar en session_state
                st.session_state.datos_procesados = True
                st.session_state.df_original = df.copy()
                st.session_state.df_dias = df_dias
                st.session_state.orden_estaciones = orden_estaciones
                
            st.success(f"✅ Datos cargados y procesados: {len(df)} registros")
        except Exception as e:
            st.error(f"Error al cargar el archivo csv: {e}")
            st.stop()
    
    # Mostrar visualizaciones (siempre que los datos estén procesados)
    if st.session_state.datos_procesados and st.session_state.df_dias is not None:
        df_dias = st.session_state.df_dias
        df_original = st.session_state.df_original
        orden_estaciones = st.session_state.orden_estaciones
        
        # ========== SELECTOR DE VISUALIZACIÓN ==========
        st.subheader("🎯 Selecciona qué información deseas explorar:")
        
        opcion = st.selectbox(
            "Elige una visualización:",
            [
                "Temperaturas promedio por mes",
                "Días de lluvia por mes",
                "Distribución de condiciones climáticas",
                "Temperatura vs sensación térmica",
                "Temperaturas extremas del año",
                "Relación humedad y temperatura",
                "Evolución de temperatura anual"
            ]
        )
        
        st.markdown("---")
        
        # ========== VISUALIZACIÓN 1: TEMPERATURAS PROMEDIO POR MES ==========
        if "Temperaturas promedio por mes" in opcion:
            st.header("📅 Temperaturas promedio por Mes")
            st.markdown("""
            **¿Qué muestra?** La temperatura promedio de cada mes del año en Mendoza.  
            **¿Para qué sirve?** Te ayuda a planificar viajes o actividades sabiendo qué meses son más calurosos o fríos.
            """)
            
            # Agrupar por mes
            df_mensual = df_dias.groupby('mes', as_index=False).agg({
                'temp_avg_dia': 'mean',
                'temp_max_dia': 'mean',
                'temp_min_dia': 'mean'
            }).round(2)
            
            # Nombres de meses
            meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            df_mensual['mes_nombre'] = df_mensual['mes'].apply(lambda x: meses_nombres[x-1])
            
            # Gráfico de barras
            chart = alt.Chart(df_mensual).mark_bar().encode(
                x=alt.X('mes_nombre:N', 
                       title='Mes',
                       sort=meses_nombres,
                       axis=alt.Axis(labelAngle=0)),
                y=alt.Y('temp_avg_dia:Q', 
                       title='Temperatura Promedio (°C)'),
                color=alt.Color('temp_avg_dia:Q',
                               scale=alt.Scale(scheme='redyellowblue', reverse=True),
                               legend=None),
                tooltip=[
                    alt.Tooltip('mes_nombre:N', title='Mes'),
                    alt.Tooltip('temp_avg_dia:Q', title='Temp. Promedio (°C)', format='.1f'),
                    alt.Tooltip('temp_max_dia:Q', title='Temp. Máx Prom (°C)', format='.1f'),
                    alt.Tooltip('temp_min_dia:Q', title='Temp. Mín Prom (°C)', format='.1f')
                ]
            ).properties(
                width=800,
                height=400,
                title='Temperatura Promedio Mensual en Mendoza'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Insight
            mes_caluroso = df_mensual.loc[df_mensual['temp_avg_dia'].idxmax()]
            mes_frio = df_mensual.loc[df_mensual['temp_avg_dia'].idxmin()]
            
            st.info(f"""
            📌 **Conclusión:**  
            - El mes **más caluroso** es **{mes_caluroso['mes_nombre']}** con {mes_caluroso['temp_avg_dia']:.1f}°C en promedio.  
            - El mes **más frío** es **{mes_frio['mes_nombre']}** con {mes_frio['temp_avg_dia']:.1f}°C en promedio.  
            - La diferencia entre el mes más caluroso y el más frío es de **{mes_caluroso['temp_avg_dia'] - mes_frio['temp_avg_dia']:.1f}°C**.
            """)
        
        # ========== VISUALIZACIÓN 2: DÍAS DE LLUVIA POR MES ==========
        elif "Días de lluvia por mes" in opcion:
            st.header("🌧️ Días de lluvia por mes")
            st.markdown("""
            **¿Qué muestra?** Cuántos días llovió en cada mes del año.  
            **¿Para qué sirve?** Ideal para planificar actividades al aire libre y evitar meses lluviosos.
            """)
            
            # Contar días lluviosos por mes
            df_lluvia_mes = df_dias[df_dias['lluvia_dia'] == True].groupby('mes').size().reset_index(name='dias_lluvia')
            
            # Completar meses sin lluvia
            todos_meses = pd.DataFrame({'mes': range(1, 13)})
            df_lluvia_mes = todos_meses.merge(df_lluvia_mes, on='mes', how='left').fillna(0)
            
            meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            df_lluvia_mes['mes_nombre'] = df_lluvia_mes['mes'].apply(lambda x: meses_nombres[x-1])
            
            # Gráfico de barras
            chart = alt.Chart(df_lluvia_mes).mark_bar(color='#3498DB').encode(
                x=alt.X('mes_nombre:N', 
                       title='Mes',
                       sort=meses_nombres,
                       axis=alt.Axis(labelAngle=0)),
                y=alt.Y('dias_lluvia:Q', 
                       title='Cantidad de Días con Lluvia'),
                tooltip=[
                    alt.Tooltip('mes_nombre:N', title='Mes'),
                    alt.Tooltip('dias_lluvia:Q', title='Días de lluvia', format='.0f')
                ]
            ).properties(
                width=800,
                height=400,
                title='Días con Lluvia por Mes en Mendoza'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Insight
            total_dias_lluvia = df_lluvia_mes['dias_lluvia'].sum()
            mes_mas_lluvioso = df_lluvia_mes.loc[df_lluvia_mes['dias_lluvia'].idxmax()]
            mes_mas_seco = df_lluvia_mes[df_lluvia_mes['dias_lluvia'] > 0].loc[df_lluvia_mes[df_lluvia_mes['dias_lluvia'] > 0]['dias_lluvia'].idxmin()] if len(df_lluvia_mes[df_lluvia_mes['dias_lluvia'] > 0]) > 0 else mes_mas_lluvioso
            
            st.info(f"""
            📌 **Conclusión:**  
            - En total llovió **{int(total_dias_lluvia)} días** durante el año registrado.  
            - El mes **más lluvioso** fue **{mes_mas_lluvioso['mes_nombre']}** con {int(mes_mas_lluvioso['dias_lluvia'])} días de lluvia.  
            - Mendoza tiene un clima predominantemente **seco**, ideal para actividades al aire libre la mayor parte del año.
            """)
        
        # ========== VISUALIZACIÓN 3: DISTRIBUCIÓN DE CONDICIONES CLIMÁTICAS ==========
        elif "Distribución de condiciones climáticas" in opcion:
            st.header("☀️ Distribución de condiciones climáticas por estación")
            st.markdown("""
            **¿Qué muestra?** La proporción de días despejados, nublados y lluviosos en cada estación del año.  
            **¿Para qué sirve?** Para entender cómo varía el clima según la estación.
            """)
            
            # Clasificar condiciones
            def clasificar_condicion(cond):
                if pd.isna(cond):
                    return 'Otro'
                cond_lower = str(cond).lower()
                if 'rain' in cond_lower or 'drizzle' in cond_lower or 'shower' in cond_lower:
                    return 'Lluvia'
                elif 'cloud' in cond_lower or 'overcast' in cond_lower:
                    return 'Nublado'
                elif 'clear' in cond_lower or 'sun' in cond_lower:
                    return 'Despejado'
                else:
                    return 'Otro'
            
            df_dias['condicion_simple'] = df_dias['conditions'].apply(clasificar_condicion)
            
            # Contar por estación
            df_condiciones = df_dias.groupby(['estacion', 'condicion_simple']).size().reset_index(name='cantidad')
            
            # Gráfico de barras apiladas
            chart = alt.Chart(df_condiciones).mark_bar().encode(
                x=alt.X('estacion:N', 
                       title='Estación del Año',
                       sort=orden_estaciones,
                       axis=alt.Axis(labelAngle=0)),
                y=alt.Y('cantidad:Q', 
                       title='Cantidad de Días'),
                color=alt.Color('condicion_simple:N',
                               title='Condición',
                               scale=alt.Scale(
                                   domain=['Despejado', 'Nublado', 'Lluvia', 'Otro'],
                                   range=['#FFD700', '#808080', '#3498DB', '#95A5A6']
                               )),
                tooltip=[
                    alt.Tooltip('estacion:N', title='Estación'),
                    alt.Tooltip('condicion_simple:N', title='Condición'),
                    alt.Tooltip('cantidad:Q', title='Días')
                ]
            ).properties(
                width=800,
                height=400,
                title='Distribución de Condiciones Climáticas por Estación'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Calcular porcentajes
            total_por_estacion = df_condiciones.groupby('estacion')['cantidad'].sum()
            df_condiciones['porcentaje'] = df_condiciones.apply(
                lambda row: (row['cantidad'] / total_por_estacion[row['estacion']]) * 100, 
                axis=1
            )
            
            st.info("""
            📌 **Conclusión:**  
            - Mendoza tiene un clima predominantemente **despejado** durante todo el año.  
            - Los días **nublados** son más frecuentes en **invierno**.  
            - La **lluvia** es más común en los meses de **verano**, aunque sigue siendo poco frecuente.
            """)
        
        # ========== VISUALIZACIÓN 4: TEMPERATURA VS SENSACIÓN TÉRMICA ==========
        elif "Temperatura vs sensación térmica" in opcion:
            st.header("🌡️ Temperatura Real vs Sensación Térmica")
            st.markdown("""
            **¿Qué muestra?** Comparación entre la temperatura real y cómo realmente se siente (sensación térmica).  
            **¿Para qué sirve?** Para entender por qué a veces hace más calor o frío de lo que indica el termómetro.
            """)
            
            # Promediar por mes
            df_feels = df_dias.groupby('mes', as_index=False).agg({
                'temp_avg_dia': 'mean',
                'feelslike_avg': 'mean'
            }).round(2)
            
            meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            df_feels['mes_nombre'] = df_feels['mes'].apply(lambda x: meses_nombres[x-1])
            
            # Preparar datos para gráfico de líneas múltiples
            df_feels_long = pd.melt(
                df_feels, 
                id_vars=['mes', 'mes_nombre'], 
                value_vars=['temp_avg_dia', 'feelslike_avg'],
                var_name='tipo',
                value_name='temperatura'
            )
            df_feels_long['tipo'] = df_feels_long['tipo'].map({
                'temp_avg_dia': 'Temperatura Real',
                'feelslike_avg': 'Sensación Térmica'
            })
            
            # Gráfico de líneas
            chart = alt.Chart(df_feels_long).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('mes_nombre:N', 
                       title='Mes',
                       sort=meses_nombres,
                       axis=alt.Axis(labelAngle=0)),
                y=alt.Y('temperatura:Q', 
                       title='Temperatura (°C)',
                       scale=alt.Scale(zero=False)),
                color=alt.Color('tipo:N',
                               title='Tipo de Medición',
                               scale=alt.Scale(
                                   domain=['Temperatura Real', 'Sensación Térmica'],
                                   range=['#E74C3C', '#F39C12']
                               )),
                tooltip=[
                    alt.Tooltip('mes_nombre:N', title='Mes'),
                    alt.Tooltip('tipo:N', title='Tipo'),
                    alt.Tooltip('temperatura:Q', title='Temperatura (°C)', format='.1f')
                ]
            ).properties(
                width=800,
                height=400,
                title='Comparación: Temperatura Real vs Sensación Térmica'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Calcular diferencia promedio
            diferencia_prom = abs(df_feels['temp_avg_dia'] - df_feels['feelslike_avg']).mean()
            
            st.info(f"""
            📌 **Conclusión:**  
            - En promedio, la **diferencia** entre temperatura real y sensación térmica es de **{diferencia_prom:.1f}°C**.  
            - La **humedad** y el **viento** son los principales factores que afectan la sensación térmica.  
            - En **verano**, la sensación térmica suele ser mayor debido a la humedad.
            """)
        
        # ========== VISUALIZACIÓN 5: TEMPERATURAS EXTREMAS ==========
        elif "Temperaturas extremas del año" in opcion:
            st.header("📊 Comparación de temperaturas extremas")
            st.markdown("""
            **¿Qué muestra?** Las temperaturas máximas y mínimas promedio de cada mes.  
            **¿Para qué sirve?** Para entender el rango de temperaturas que puedes esperar en cada época del año.
            """)
            
            # Agrupar por mes
            df_extremos = df_dias.groupby('mes', as_index=False).agg({
                'temp_max_dia': 'mean',
                'temp_min_dia': 'mean'
            }).round(2)
            
            meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            df_extremos['mes_nombre'] = df_extremos['mes'].apply(lambda x: meses_nombres[x-1])
            
            # Preparar datos para gráfico
            df_extremos_long = pd.melt(
                df_extremos,
                id_vars=['mes', 'mes_nombre'],
                value_vars=['temp_max_dia', 'temp_min_dia'],
                var_name='tipo',
                value_name='temperatura'
            )
            df_extremos_long['tipo'] = df_extremos_long['tipo'].map({
                'temp_max_dia': 'Temperatura Máxima',
                'temp_min_dia': 'Temperatura Mínima'
            })
            
            # Gráfico de barras agrupadas
            chart = alt.Chart(df_extremos_long).mark_bar().encode(
                x=alt.X('mes_nombre:N', 
                       title='Mes',
                       sort=meses_nombres,
                       axis=alt.Axis(labelAngle=0)),
                y=alt.Y('temperatura:Q', 
                       title='Temperatura (°C)'),
                color=alt.Color('tipo:N',
                               title='Tipo',
                               scale=alt.Scale(
                                   domain=['Temperatura Máxima', 'Temperatura Mínima'],
                                   range=['#E74C3C', '#3498DB']
                               )),
                xOffset='tipo:N',
                tooltip=[
                    alt.Tooltip('mes_nombre:N', title='Mes'),
                    alt.Tooltip('tipo:N', title='Tipo'),
                    alt.Tooltip('temperatura:Q', title='Temperatura (°C)', format='.1f')
                ]
            ).properties(
                width=800,
                height=400,
                title='Temperaturas Máximas y Mínimas Promedio por Mes'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Calcular amplitud térmica
            df_extremos['amplitud'] = df_extremos['temp_max_dia'] - df_extremos['temp_min_dia']
            mes_mayor_amplitud = df_extremos.loc[df_extremos['amplitud'].idxmax()]
            
            st.info(f"""
            📌 **Conclusión:**  
            - El mes con **mayor amplitud térmica** es **{meses_nombres[mes_mayor_amplitud['mes']-1]}** con {mes_mayor_amplitud['amplitud']:.1f}°C de diferencia entre máxima y mínima.  
            - Mendoza tiene un clima con **amplitudes térmicas significativas**, especialmente en primavera y otoño.  
            - Es importante llevar ropa **adecuada para cambios de temperatura** durante el día.
            """)
        
        # ========== VISUALIZACIÓN 6: HUMEDAD VS TEMPERATURA ==========
        elif "6. Relación humedad y temperatura" in opcion:
            st.header("💧 Relación entre humedad y temperatura")
            st.markdown("""
            **¿Qué muestra?** Cómo se relaciona la humedad con la temperatura en diferentes estaciones.  
            **¿Para qué sirve?** Para entender por qué algunos días calurosos se sienten más "pesados" que otros.
            """)
            
            # Tomar muestra para mejor visualización
            df_sample = df_dias.sample(min(500, len(df_dias)))
            
            # Gráfico de dispersión
            chart = alt.Chart(df_sample).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('temp_avg_dia:Q', 
                       title='Temperatura Promedio (°C)'),
                y=alt.Y('humidity_avg:Q', 
                       title='Humedad Promedio (%)'),
                color=alt.Color('estacion:N',
                               title='Estación',
                               scale=alt.Scale(
                                   domain=orden_estaciones,
                                   range=['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
                               )),
                tooltip=[
                    alt.Tooltip('dia:T', title='Fecha', format='%Y-%m-%d'),
                    alt.Tooltip('temp_avg_dia:Q', title='Temperatura (°C)', format='.1f'),
                    alt.Tooltip('humidity_avg:Q', title='Humedad (%)', format='.1f'),
                    alt.Tooltip('estacion:N', title='Estación')
                ]
            ).properties(
                width=800,
                height=400,
                title='Relación entre Temperatura y Humedad por Estación'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            st.info("""
            📌 **Conclusión:**  
            - En **verano**, la combinación de alta temperatura y humedad genera una sensación térmica más elevada.  
            - En **invierno**, la baja humedad hace que el frío se sienta más seco y penetrante.  
            - La humedad promedio en Mendoza es relativamente **baja** comparada con otras regiones de Argentina.
            """)
        
        # ========== VISUALIZACIÓN 7: EVOLUCIÓN ANUAL ==========
        elif "Evolución de temperatura anual" in opcion:
            st.header("📈 Evolución de la temperatura durante el año")
            st.markdown("""
            **¿Qué muestra?** Cómo varía la temperatura día a día a lo largo del año.  
            **¿Para qué sirve?** Para visualizar claramente las cuatro estaciones y sus transiciones.
            """)
            
            # Ordenar por fecha
            df_evolucion = df_dias.sort_values('dia').copy()
            df_evolucion['dia_año'] = df_evolucion['dia'].dt.dayofyear
            
            # Crear gráfico de área
            base = alt.Chart(df_evolucion).encode(
                x=alt.X('dia:T', 
                       title='Fecha',
                       axis=alt.Axis(format='%b')),
            )
            
            # Área para rango min-max
            area = base.mark_area(opacity=0.3, color='#95A5A6').encode(
                y=alt.Y('temp_min_dia:Q', title='Temperatura (°C)'),
                y2='temp_max_dia:Q'
            )
            
            # Línea para temperatura promedio
            line = base.mark_line(color='#E74C3C', strokeWidth=2).encode(
                y=alt.Y('temp_avg_dia:Q', title='Temperatura (°C)'),
                tooltip=[
                    alt.Tooltip('dia:T', title='Fecha', format='%Y-%m-%d'),
                    alt.Tooltip('temp_avg_dia:Q', title='Temp. Promedio (°C)', format='.1f'),
                    alt.Tooltip('temp_max_dia:Q', title='Temp. Máxima (°C)', format='.1f'),
                    alt.Tooltip('temp_min_dia:Q', title='Temp. Mínima (°C)', format='.1f'),
                    alt.Tooltip('estacion:N', title='Estación')
                ]
            )
            
            chart = (area + line).properties(
                width=800,
                height=400,
                title='Evolución de la Temperatura en Mendoza'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            st.info("""
            📌 **Conclusión:**  
            - Se observa claramente el patrón de las **cuatro estaciones**.  
            - La transición de **invierno a verano** es más gradual que la de verano a invierno.  
            - El área sombreada muestra la **amplitud térmica diaria** (diferencia entre máxima y mínima).
            """)
        
        # ========== SECCIÓN ADICIONAL: DATOS CRUDOS ==========
        st.markdown("---")
        with st.expander("📋 Ver datos completos en tabla"):
            st.subheader("Datos agregados por día")
            st.dataframe(
                df_dias[['dia', 'estacion', 'temp_max_dia', 'temp_min_dia', 
                        'temp_avg_dia', 'humidity_avg', 'condicion_dia', 'conditions']].sort_values('dia', ascending=False),
                use_container_width=True
            )
            
            # Botón de descarga
            csv = df_dias.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar datos como CSV",
                data=csv,
                file_name='datos_clima_mendoza.csv',
                mime='text/csv',
            )