import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta

# Load data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['fecha'], dayfirst=True)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df.set_index('fecha', inplace=True)
    return df

# File Path
data_path = './rama_2023_05 (1).csv'
df = load_data(data_path)

# Streamlit App
st.title("Calidad del Aire en la CDMX (2015-2023)")
st.markdown("""
    La Ciudad de México es la capital y la ciudad más grande de México, así como también es la ciudad más poblada de Norteamérica. Se localiza en el valle de México en la región centro sur del país a una altura promedio de 2240 sobre el nivel del mar. Debido a que se trata de una de las zonas con mayor mancha urbana en el país, se enfrenta a grandes problemas ecólogicos, como la contaminación y el desabasto de agua. 
    
    Durante las últimas décadas han habido esfuerzos para mitigar esta situación, como es el caso del sistema IMECA y del programa hoy no circula. En ambos casos, se utlizan los datos generados por mediciones por la Red Autómatica de Monitoreo Atmósferico (RAMA), que cuenta con 34 estaciones repartidas en las alcaldias de la CDMX así como en algunas municipios conurbanos del Estado de México. 
  
 """)

# Sidebar
st.sidebar.title("Acerca de los datos")
st.sidebar.markdown("""
    Los datos fueron descargados de [datos.cdmx.gob.mx](https://datos.cdmx.gob.mx/dataset/red-automatica-de-monitoreo-atmosferico),
    contienen las mediciones tomadas por la Red Autómatica de Monitoreo Atmósferico (RAMA), el archivo csv descargado contiene los 
    promedios diarios de los registros de varias particulas para la Ciudad de México desde el 2015.
    
    """)
# Selection menu
pollutants = df.columns.tolist()
all_option = ["Todos"] + pollutants
selected_pollutants = st.sidebar.multiselect("**Selecciona que contaminantes deseas ver**", options=all_option, default=["Todos"])
st.sidebar.markdown("""
    Abreviaciones:
    
    - CO: Monóxido de Carbono
    - NO: Monóxido de Nitrógeno
    - NO2: Dióxido de Nitrógeno
    - NOX: Oxidos de Nitrógeno
    - O3: Ozono
    - PM25: Párticulas menores a 2.5 micras
    - PM10: Párticulas menores a 10 micras
    - Dióxido de Azufre
    
    **Nota**: Las gráficas pueden tardar un poco en cargar debido al cálculo que debe realizarse para mostrar 
    predicciones en la gráfica de la tendencia mensual
""")

# 
if "Todos" in selected_pollutants:
    selected_pollutants = pollutants
elif not selected_pollutants:
    selected_pollutants = [] 

# RAW
st.subheader("Datos Crudos")
st.markdown("""Los datos fueron obtenidos del portal datos.cdmx.gob.mx, el archivo contiene los valores promedios por día de diferentes contaminantes.
Estos datos están preprocesados, al momento de este análisis no fue posible conseguir los datos por hora y por estación de la RAMA en el portal de aire.cdmx.gob.mx.

A continuación se muestra la forma que tiene el archivo disponible:
""")
st.write(df.head())


##################################
st.subheader("Casos de interes: Pandemia COVID-19")
st.markdown(""" A incicios de la pandemia de COVID-19, se implementaron restricciones de movilidad las cuales tuvieron un efecto positivo en la calidad del aire. Debido reducción de los autos en circulación así como también una reducción de actividades industriales fue posible observar una menor cantidad de contaminantes en la zona del valle de México.

En las gráficas generadas se puede confirmar que las mediciones realizadas durante la pandemia mostraron menor cantidad de contaminantes, sin embargo después de que las restricciones fueron levantadas los niveles de contaminación volvieron a aumentar. 
El único contaminante con mayor presencia durante la pandemia se trató del Ozono, sin embargo su presencia elevada puede estar relacionada a periodos con ondas de calor, se necesitaria realizar un análisis junto con mediciones de temperatura para confirmar esta hipótesis.
""")


############################
st.subheader("Hoy no circula")
st.markdown(""" El programa hoy no circula fue implementado desde 1989 como medida para reducir los niveles de contaminación al limitar la cantidad de autos en circulación. Inicialmente fue concebido para limitar el 20% de los autos, sin embargo la cantidad de autos restrigidos es cerca del 7%. En el 2008 se agregaron dos sábados al mes a las restricciones del programa. 

En las gráficas de temporalidad por semana, se puede observar que no hay un cambio significativo en las mediciones de contaminanantes en los dias sábado y domingo, lo cual pone en duda la efectividad del programa hoy no circula.
""")


##################################
st.subheader("Análisis por contaminante")
st.markdown(""" Enseguida se muestran las gráficas generadas a partir de los datos de la calidad del aire en la CDMX del periodo 2015 al 2023. 

La primera gráfica es una serie de tiempo con los promedios diarios y el promedio mensual de los contaminantes. En la primera gráfica también se incluye un pronóstico de las mediciones de contaminantes del 2024. El pronóstico fue realizado con Modelo autorregresivo integrado de media móvil (ARIMA).

La segunda gráfica muestra la temporalidad estacional de cada contaminante, es posible observar ciclos estacionales, con mayor cantidad de contaminantes como el CO, NO y NO2 en los meses de invierno, mientras que las concentraciones de ozono siguen un ciclo contrario, con mayor concentración en el verano.

La tercera gráfica muestra la distribución de las concentraciones de los contaminantes por dia. Aqui puede observarse que no hay cambios significativos entre los dias de la semana, solo una pequeña disminución los fines de semana.

La cuarta gráfica es la densidad de distribución en la cual se puede apreciar con mayor facilidad la reducción de contaminantes durantes la pandemia.
""")

# Correlation Matrix
#st.subheader("Matriz de correlación")
#corr_matrix = df[pollutants].corr()
# Plot correlation matrix as a heatmap
#fig, ax = plt.subplots(figsize=(10, 8))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
#ax.set_title("Correlation Matrix of Pollutants")
#st.pyplot(fig)

# Data prep
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Weekday'] = df.index.weekday
monthly_avg = df.resample('ME').mean()
yearly_seasonal = df.groupby(['Year', 'Month']).mean().unstack(level=0)
specific_years = [2020, 2021, 2023]

# Dic for months
month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# Forecast config
forecast_periods = 12  # Number of months to forecast (one year)

# Plot for each pollutant
for pollutant in selected_pollutants:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    #fig.suptitle(f'{pollutant} Analysis', fontsize=16)
    fig.suptitle(f'Análisis de {pollutant}', fontsize=16)
    
    # 1. Daily Data with Monthly Trend and Forecast
    ax = axes[0, 0]
    ax.plot(df.index, df[pollutant], color='lightgray', alpha=0.5, label='Promedio diario')
    ax.plot(monthly_avg.index, monthly_avg[pollutant], color='blue', label='Tendencia mensual')

    # Prepare data for forecasting
    df_resampled = df[pollutant].resample('ME').mean()
    
    result = adfuller(df_resampled.dropna())
    st.write(f"{pollutant} - ADF Statistic: {result[0]}, p-value: {result[1]}")
    
    if result[1] > 0.05:  # If p-value > 0.05
        df_resampled_diff = df_resampled.diff().dropna()
    else:
        df_resampled_diff = df_resampled
    
    # Fit ARIMA
    try:
        model = auto_arima(df_resampled_diff, seasonal=True, m=12, stepwise=True, trace=True)
        
        # forecast
        forecast = model.predict(n_periods=forecast_periods)
        forecast_index = pd.date_range(df.index[-1] + timedelta(days=1), periods=forecast_periods, freq='ME')
        
        if result[1] > 0.05:
            forecast = forecast.cumsum() + df_resampled.iloc[-1] 

        # Plot forecast
        ax.plot(forecast_index, forecast, color='red', linestyle='--', label='Pronóstico 2024')
    except Exception as e:
        st.write(f"ARIMA model failed for {pollutant} due to: {e}")

    ax.set_title("Datos diarios con tenedencia mensual y pronóstico")
    ax.set_ylabel(f"{pollutant} ppm")
    ax.legend()

    # 2. Monthly Season
    ax = axes[0, 1]
    max_values = yearly_seasonal[pollutant].max(axis=1)
    min_values = yearly_seasonal[pollutant].min(axis=1)
    avg_values = yearly_seasonal[pollutant].mean(axis=1)

    ax.fill_between(yearly_seasonal.index, min_values, max_values, color='skyblue', alpha=0.5, label='Min-Max')
    ax.plot(yearly_seasonal.index, avg_values, color='black', label='Promedio', linewidth=2)
    if 2020 in yearly_seasonal[pollutant].columns:
        ax.plot(yearly_seasonal.index, yearly_seasonal[pollutant][2020], color='red', linestyle='--', label='2020', linewidth=2)
    ax.set_title("Temporalidad mensual")
    ax.set_ylabel(f"{pollutant} cpp")
    ax.legend(loc='upper right', fontsize='small')

    # Set month names as x-axis labels
    ax.set_xticks(range(1, 13))  # Set ticks for each month
    ax.set_xticklabels(month_names)  # Replace tick labels with month names

    # 3. Weekday Seasonality as Box Plot
    ax = axes[1, 0]
    sns.boxplot(data=df, x='Weekday', y=pollutant, ax=ax, palette="Set3")
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'])
    ax.set_title("Distribución por dia de la semana")
    ax.set_ylabel(f"{pollutant} Level")

    # 4. Density Distribution (Specific Years)
    ax = axes[1, 1]
    sns.kdeplot(data=df, x=pollutant, fill=True, ax=ax, color="skyblue", linewidth=1.5, label="Total")
    for year in specific_years:
        sns.kdeplot(data=df[df['Year'] == year], x=pollutant, ax=ax, linewidth=1.5, label=str(year))
    ax.set_title("Distribución de Densidad")
    ax.set_xlabel(f"Nivel de {pollutant}")
    ax.set_ylabel("Densidad")
    ax.legend(title="Año", loc="upper right", fontsize=8)

    st.pyplot(fig)
