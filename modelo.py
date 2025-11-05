import os
import poplib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


# ---------------------------
# Función para reducir condiciones (a 4 clases base)
# ---------------------------
def resumir_target_v3(lista_conditions):
    if isinstance(lista_conditions, (str, float, int)) and not isinstance(lista_conditions, list):
        lista_conditions = [lista_conditions]
    lista_conditions = [str(x).strip() for x in lista_conditions if pd.notna(x)]
    joined = ", ".join(lista_conditions)
    if "Rain" in joined:
        return "Rain"
    opciones = ["Partially cloudy", "Overcast", "Clear"]
    contadas = [c for c in lista_conditions if c in opciones]
    if contadas:
        return Counter(contadas).most_common(1)[0][0]
    else:
        return "Clear"

# ---------------------------
# 1) Cargar dataset original
# ---------------------------
df = pd.read_csv("joined_weather_data.csv")
print("Dataset original cargado:", df.shape)

if 'datetime_completo' not in df.columns:
    raise KeyError("Falta la columna 'datetime_completo' en el CSV.")
df['datetime_completo'] = pd.to_datetime(df['datetime_completo'], errors='coerce')

if 'dia' in df.columns:
    df['dia'] = pd.to_datetime(df['dia'], errors='coerce')
else:
    df['dia'] = df['datetime_completo'].dt.floor('d')

# ---------------------------
# 2) Feature auxiliar horaria
# ---------------------------
if 'temp' in df.columns and 'dew' in df.columns:
    df['hourly_dew_diff'] = df['temp'] - df['dew']
else:
    df['hourly_dew_diff'] = np.nan

# ---------------------------
# 3) Agregación diaria
# ---------------------------
agg_dict = {
    'temp': ['mean', 'max', 'min'],
    'feelslike': 'mean',
    'humidity': 'mean',
    'dew': 'mean',
    'hourly_dew_diff': 'mean',
    'precip': 'sum',
    'precipprob': 'mean',
    'snow': 'sum',
    'snowdepth': 'max',
    'windgust': 'mean',
    'windspeed': 'mean',
    'winddir': 'mean',
    'pressure': 'mean',
    'visibility': 'mean',
    'cloudcover': 'mean',
    'solarradiation': 'mean',
    'solarenergy': 'mean',
    'uvindex': 'mean',
    'conditions': lambda x: list(x)
}
agg_dict = {k:v for k,v in agg_dict.items() if k in df.columns}

df_daily = df.groupby(df['dia']).agg(agg_dict)
df_daily.columns = [
    f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col)
    for col in df_daily.columns.to_flat_index()
]
df_daily = df_daily.reset_index().rename(columns={'dia':'date'})

# ---------------------------
# 4) Resumir condiciones
# ---------------------------
cond_col = [c for c in df_daily.columns if c.startswith('conditions')][0]
df_daily['conditions_reduced'] = df_daily[cond_col].apply(resumir_target_v3)
df_daily = df_daily.drop(columns=[cond_col])

# ---------------------------
# 5) Crear target desplazado (día siguiente)
# ---------------------------
df_daily['target'] = df_daily['conditions_reduced'].shift(-1)
df_daily = df_daily.iloc[:-1].copy()

# ---------------------------
# 6) Features derivadas
# ---------------------------
df_daily['temp_range'] = df_daily['temp_max'] - df_daily['temp_min']
df_daily['dew_point_diff'] = df_daily['temp_mean'] - df_daily['dew_mean']

# ---------------------------
# 7) Estacionalidad y lluvia previa
# ---------------------------
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily['month'] = df_daily['date'].dt.month
df_daily['dayofyear'] = df_daily['date'].dt.dayofyear
df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['month'] / 12)
df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['month'] / 12)
df_daily['dayofyear_sin'] = np.sin(2 * np.pi * df_daily['dayofyear'] / 365)
df_daily['dayofyear_cos'] = np.cos(2 * np.pi * df_daily['dayofyear'] / 365)
df_daily['rained_today'] = (df_daily['conditions_reduced'] == 'Rain').astype(int)
df_daily['rain_yesterday'] = df_daily['rained_today'].shift(1).fillna(0).astype(int)

# ---------------------------
# 8) Fusionar "Partially cloudy" + "Overcast" → "Cloudy"
# ---------------------------
df_daily['conditions_reduced'] = df_daily['conditions_reduced'].replace({
    'Partially cloudy': 'Cloudy',
    'Overcast': 'Cloudy'
})
df_daily['target'] = df_daily['target'].replace({
    'Partially cloudy': 'Cloudy',
    'Overcast': 'Cloudy'
})

# ---------------------------
# 9) Features finales
# ---------------------------
NUM_FEATS = [
    'temp_mean','feelslike_mean','humidity_mean','dew_mean','pressure_mean',
    'windspeed_mean','windgust_mean','winddir_mean','visibility_mean',
    'solarradiation_mean','uvindex_mean','cloudcover_mean','precip_sum','snow_sum',
    'temp_range','dew_point_diff','month_sin','month_cos','dayofyear_sin','dayofyear_cos'
]
NUM_FEATS = [c for c in NUM_FEATS if c in df_daily.columns]
CAT_FEATS = ['rain_yesterday']

df_daily = df_daily.dropna(subset=['target'])
X = df_daily[NUM_FEATS + CAT_FEATS].copy()
y = df_daily['target'].copy()

print("\nDataset diario procesado (primeras filas):")
print(df_daily.head()[NUM_FEATS + CAT_FEATS + ['target']])

# ---------------------------
# 10) Split aleatorio estratificado
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nDistribución de clases en Train:")
print(y_train.value_counts(normalize=True).round(3))
print("\nDistribución de clases en Test:")
print(y_test.value_counts(normalize=True).round(3))

# ---------------------------
# 11) Preprocesador
# ---------------------------
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, NUM_FEATS),
    ('cat', cat_transformer, CAT_FEATS)
])

# ---------------------------
# 12) Modelo único: Gradient Boosting
# ---------------------------
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', gb_model)])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1w = f1_score(y_test, y_pred, average='weighted')
f1m = f1_score(y_test, y_pred, average='macro')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nResumen de métricas finales:")
print(pd.DataFrame([{
    'Modelo': 'Gradient Boosting',
    'Accuracy': acc,
    'F1 weighted': f1w,
    'F1 macro': f1m
}]))

# ---------------------------
# 13) Guardar modelo entrenado
# ---------------------------
output_dir = "model_output"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "gradient_boosting_weather_model.pkl")
poplib.dump(pipeline, model_path)

print(f"\n✅ Modelo guardado correctamente en: {model_path}")
