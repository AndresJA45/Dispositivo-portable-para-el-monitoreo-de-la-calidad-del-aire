import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os

# Cargar los datos desde el archivo Excel
data = pd.read_excel(r'C:\Users\Usuario-PC\Desktop\TESIS AUNAR\BASE_DATOS2.xlsx', sheet_name='MES2NP')

# Preprocesamiento de los datos
X = data[['CO2','CO', 'O3', 'NO2', 'PM2.5', 'PM10']]
y = data[['CO2', 'CO', 'O3', 'NO2', 'PM2.5', 'PM10']]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba.
X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y_normalized, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Construcción del modelo de red neuronal
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_normalized.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_normalized.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Realización del pronóstico en el conjunto de prueba
y_pred_normalized = model.predict(X_test)

# Deshacer la normalización para obtener los valores reales del pronóstico
y_pred = scaler_y.inverse_transform(y_pred_normalized)
y_test_original = scaler_y.inverse_transform(y_test)

# Calcular el promedio de cada valor de las matrices
variables = ['CO2', 'CO', 'O3', 'NO2', 'PM2.5', 'PM10']
averages = {}
for i, variable in enumerate(variables):
    real_values = y_test_original[:, i]
    pred_values = y_pred[:, i]
    
    # Calcular el promedio de los valores reales y de predicción
    avg_real = np.mean(real_values)
    avg_pred = np.mean(pred_values)
    
    averages[f'{variable}_Real'] = avg_real
    averages[f'{variable}_Pred'] = avg_pred


# Convertir los promedios a un DataFrame
averages_df = pd.DataFrame(averages, index=[0])

# Exportar a un archivo Excel y guardar los promedios
file_path = r'C:\Users\Usuario-PC\Desktop\TESIS AUNAR\ResultadosNUEVOS.xlsx'
if os.path.exists(file_path):
    # Leer el archivo Excel existente
    existing_data = pd.read_excel(file_path)
    
    # Concatenar los nuevos promedios al DataFrame existente
    all_data = pd.concat([existing_data, averages_df], ignore_index=True)
    all_data.to_excel(file_path, index=False)
else:
    averages_df.to_excel(file_path, index=False)