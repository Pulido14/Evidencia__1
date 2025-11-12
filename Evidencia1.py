import pandas as pd
import numpy as np

# --- 4.8. Carga y Limpieza Inicial (Asumiendo que ya se hizo, pero se incluye para que corra) ---

# Cargar el archivo
try:
    df = pd.read_csv('ShoeLand_Master_Dirty(in).csv')
except FileNotFoundError:
    print("Error: Asegúrate de que el archivo 'ShoeLand_Master_Dirty(in).csv' esté en la misma carpeta.")
    exit()

# Renombrar columnas clave (basado en el análisis previo del archivo original)
df.rename(columns={
    'cantidad': 'cantidad_item',
    'utilidad': 'utilidad_item',
    'id_tipo_calzado': 'id_tipo_calzado_original'
}, inplace=True)

# Conversión de tipos de datos (como se hizo previamente)
df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
df['tipo_calzado'] = df['tipo_calzado'].astype('category')
df['pais'] = df['pais'].astype('category')
df['medida_item'] = df['medida_item'].replace(0, np.nan) # Asume 0 es nulo
df.drop_duplicates(inplace=True)

# --------------------------------------------------------------------------
# --- 4.8.6. Manejo de datos ausentes (medida_item) ---
# --------------------------------------------------------------------------

# Imputación de ausentes en medida_item (talla) con la moda
moda_medida = df['medida_item'].mode()[0]
df['medida_item'].fillna(moda_medida, inplace=True)
df['medida_item'] = df['medida_item'].astype(int)
print("\n[4.8.6] Conteo de valores nulos después de imputación:")
print(df[['medida_item', 'venta_item', 'utilidad_item']].isnull().sum())


# --------------------------------------------------------------------------
# --- 4.8.1. Manejo de datos atípicos (Outliers) ---
# --------------------------------------------------------------------------

def cap_outliers_iqr(df_in, col_name):
    Q1 = df_in[col_name].quantile(0.25)
    Q3 = df_in[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_out = df_in.copy()
    # Aplicar capping (limitar valores fuera de los límites)
    df_out[col_name] = np.where(df_out[col_name] < lower_bound, lower_bound, df_out[col_name])
    df_out[col_name] = np.where(df_out[col_name] > upper_bound, upper_bound, df_out[col_name])
    return df_out

# Aplicar el método IQR a las variables cuantitativas clave
df = cap_outliers_iqr(df, 'venta_item')
df = cap_outliers_iqr(df, 'utilidad_item')
df = cap_outliers_iqr(df, 'cantidad_item')
print("\n[4.8.1] Manejo de Atípicos: Variables cuantitativas tratadas con método IQR.")


# --------------------------------------------------------------------------
# --- 4.8.2. Muestreo Estratificado ---
# --------------------------------------------------------------------------

# Generar una muestra estratificada del 10% por tipo_calzado
FRAC_MUESTRA = 0.10
df_muestra = df.groupby('tipo_calzado', group_keys=False).apply(
    lambda x: x.sample(frac=FRAC_MUESTRA, random_state=42)
)
print(f"\n[4.8.2] Muestra Estratificada generada (10%): {df_muestra.shape[0]} filas.")


# --------------------------------------------------------------------------
# --- 5.1. Análisis de Parámetros (5 Incógnitas - Usando df_muestra) ---
# --------------------------------------------------------------------------

print("\n\n--- 5.1. ANÁLISIS DE PARÁMETROS ---")

# 1. Mediana (Tendencia Central)
mediana_venta = df_muestra['venta_item'].median()
print(f"1. Mediana de venta_item: ${mediana_venta:,.2f}")

# 2. Desviación Estándar (Dispersión)
std_utilidad = df_muestra['utilidad_item'].std()
print(f"2. Desviación Estándar de utilidad_item: ${std_utilidad:,.2f}")

# 3. Coeficiente de Variación (Dispersión Relativa)
cv_venta = df_muestra['venta_item'].std() / df_muestra['venta_item'].mean()
print(f"3. Coeficiente de Variación (CV) para venta_item: {cv_venta:.2f} ({(cv_venta*100):.2f}%)")

# 4. Mediana de cantidad_item por Tipo de Calzado
mediana_cantidad_por_tipo = df_muestra.groupby('tipo_calzado')['cantidad_item'].median().sort_values(ascending=False)
print("\n4. Mediana de cantidad_item por tipo_calzado:")
print(mediana_cantidad_por_tipo)

# 5. Varianza de la utilidad_item
varianza_utilidad = df_muestra['utilidad_item'].var()
print(f"\n5. Varianza de la utilidad_item: {varianza_utilidad:,.2f}")


# --------------------------------------------------------------------------
# --- 5.2. Análisis de Frecuencias (5 Incógnitas - Usando df_muestra) ---
# --------------------------------------------------------------------------

print("\n\n--- 5.2. ANÁLISIS DE FRECUENCIAS ---")

# 1. Frecuencia de tipo_calzado (por transacciones)
frecuencia_tipo = df_muestra['tipo_calzado'].value_counts()
print("\n1. Frecuencia Absoluta (Transacciones) por tipo_calzado:")
print(frecuencia_tipo)

# 2. Frecuencia de medida_item (talla)
frecuencia_talla = df_muestra['medida_item'].value_counts().head(5)
print("\n2. Top 5 de Frecuencia Absoluta por medida_item (Talla):")
print(frecuencia_talla)

# 3. Frecuencia de cantidad_item por transacción (para validar H2)
frecuencia_cantidad = df_muestra['cantidad_item'].value_counts(normalize=True).round(4) * 100
print("\n3. Frecuencia Relativa (%) de cantidad_item por transacción:")
print(frecuencia_cantidad.head(5))

# 4. Frecuencia por País
frecuencia_pais = df_muestra['pais'].value_counts()
print("\n4. Frecuencia Absoluta (Transacciones) por País:")
print(frecuencia_pais)

# 5. Frecuencia Relativa de los 3 principales tipo_calzado
frecuencia_relativa_top3 = df_muestra['tipo_calzado'].value_counts(normalize=True).head(3).round(4) * 100
print("\n5. Frecuencia Relativa (%) de los 3 principales tipo_calzado:")
print(frecuencia_relativa_top3)


# --------------------------------------------------------------------------
# --- 5.3.2. Análisis de Contingencia (5 Incógnitas - Usando df_muestra) ---
# --------------------------------------------------------------------------

print("\n\n--- 5.3.2. ANÁLISIS DE CONTINGENCIA O DOBLE CATEGÓRICO ---")

# 1. Venta Total por tipo_calzado vs País
venta_total_contingencia = df_muestra.pivot_table(
    index='tipo_calzado',
    columns='pais',
    values='venta_item',
    aggfunc='sum'
).round(2)
print("\n1. Venta Total por tipo_calzado vs País:")
print(venta_total_contingencia)

# 2. Utilidad Promedio por tipo_calzado vs País (Para validar H2: Rentabilidad vs Ubicación)
utilidad_promedio_contingencia = df_muestra.pivot_table(
    index='tipo_calzado',
    columns='pais',
    values='utilidad_item',
    aggfunc='mean'
).round(2)
print("\n2. Utilidad Promedio por tipo_calzado vs País:")
print(utilidad_promedio_contingencia)

# 3. Cantidad Total (Volumen) por tipo_calzado vs Talla (medida_item)
cantidad_volumen_contingencia = df_muestra.pivot_table(
    index='tipo_calzado',
    columns='medida_item',
    values='cantidad_item',
    aggfunc='sum',
    fill_value=0
).iloc[:, 0:5] # Mostrar solo las primeras 5 tallas para mejor visualización
print("\n3. Cantidad Total por tipo_calzado vs Talla (Top 5 Tallas):")
print(cantidad_volumen_contingencia)

# 4. Correlación entre Venta y Utilidad (Para validar H2/H3: ¿El precio se relaciona con la utilidad?)
correlacion_utilidad_venta = df_muestra[['venta_item', 'utilidad_item']].corr()
print("\n4. Coeficiente de Correlación entre Venta y Utilidad:")
print(correlacion_utilidad_venta)

# 5. Utilidad Total por tipo_calzado vs Local
utilidad_total_local = df_muestra.pivot_table(
    index='tipo_calzado',
    columns='local_id',
    values='utilidad_item',
    aggfunc='sum',
    fill_value=0
).iloc[:, 0:3] # Mostrar solo los primeros 3 locales
print("\n5. Utilidad Total por tipo_calzado vs Local (Top 3 Locales):")
print(utilidad_total_local)
