# generar_dataset.py

import numpy as np
import pandas as pd

# Parámetros del experimento
N = 1400  # Número de iteraciones (jugadas)
d = 12    # Número de máquinas tragamonedas

# Generar probabilidades aleatorias para cada máquina
# Estas probabilidades representan la tasa de éxito de cada máquina
true_probabilities = np.random.uniform(0, 1, d)

# Generar el dataset
dataset = np.zeros((N, d))

for i in range(d):
    # Generar resultados binarios (0 o 1) según la probabilidad de cada máquina
    dataset[:, i] = np.random.binomial(1, true_probabilities[i], N)

# Convertir el array a DataFrame de pandas
dataset_df = pd.DataFrame(dataset, columns=[f'Máquina_{i+1}' for i in range(d)])

# Guardar el dataset en un archivo CSV
dataset_df.to_csv('maquinas_tragamonedas.csv', index=False)

print("Dataset 'maquinas_tragamonedas.csv' generado exitosamente.")
