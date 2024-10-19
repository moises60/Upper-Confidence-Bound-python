import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Cargar el dataset
dataset = pd.read_csv("maquinas_tragamonedas.csv")

# Implementación del Algoritmo UCB
N = len(dataset)
d = len(dataset.columns)
number_of_selections = [0] * d
sums_of_rewards = [0] * d
machines_selected = []
total_reward = 0

# Inicialización para rastrear recompensas por máquina en cada iteración
rewards_matrix = np.zeros((N, d))

for n in range(0, N):
    max_upper_bound = 0
    machine = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400  # Valor muy alto para asegurar la selección inicial
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            machine = i
    machines_selected.append(machine)
    number_of_selections[machine] += 1
    reward = dataset.values[n, machine]
    sums_of_rewards[machine] += reward
    total_reward += reward
    
    # Registrar la recompensa obtenida por la máquina seleccionada en esta iteración
    rewards_matrix[n, machine] = reward

# Visualización de resultados - Histograma de selecciones
plt.figure(figsize=(12,6))
plt.hist(machines_selected, bins=np.arange(d+1)-0.5, edgecolor='black')
plt.title("Selecciones de Máquinas por el Algoritmo UCB")
plt.xlabel("ID de la Máquina")
plt.ylabel("Número de veces que fue seleccionada")
plt.xticks(range(d))
plt.show()

# Visualización de la recompensa acumulada
rewards_per_iteration = np.cumsum([dataset.values[n, machines_selected[n]] for n in range(N)])
plt.figure(figsize=(12,6))
plt.plot(range(N), rewards_per_iteration)
plt.title("Recompensa Acumulada a lo Largo del Tiempo")
plt.xlabel("Número de Iteraciones")
plt.ylabel("Recompensa Acumulada")
plt.show()

# Comparación con selección aleatoria
import random
total_reward_random = 0
machines_selected_random = []
rewards_matrix_random = np.zeros((N, d))

for n in range(0, N):
    machine = random.randrange(d)
    machines_selected_random.append(machine)
    reward = dataset.values[n, machine]
    total_reward_random += reward
    rewards_matrix_random[n, machine] = reward

# Visualización comparativa de recompensas
plt.figure(figsize=(12,6))
plt.plot(range(N), rewards_per_iteration, label='UCB')
rewards_random = np.cumsum([dataset.values[n, machines_selected_random[n]] for n in range(N)])
plt.plot(range(N), rewards_random, label='Selección Aleatoria')
plt.title("Comparación de Recompensas Acumuladas")
plt.xlabel("Número de Iteraciones")
plt.ylabel("Recompensa Acumulada")
plt.legend()
plt.show()

print(f"Recompensa total obtenida por UCB: {total_reward}")
print(f"Recompensa total obtenida por selección aleatoria: {total_reward_random}")


# Calcular la recompensa acumulada para cada máquina a lo largo de las iteraciones
cumulative_rewards = np.cumsum(rewards_matrix, axis=0)

# Visualización de las recompensas acumuladas por cada máquina
plt.figure(figsize=(14, 8))

for i in range(d):
    plt.plot(cumulative_rewards[:, i], label=f'Máquina {i}')

plt.title("Evolución de las Recompensas Acumuladas por Máquina")
plt.xlabel("Número de Iteraciones")
plt.ylabel("Recompensa Acumulada")
plt.legend()
plt.grid(True)
plt.show()
