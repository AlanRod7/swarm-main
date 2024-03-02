# Experimento TOPSIS
#
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# ACtualizado Feb-2024

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import datetime
import asyncio



hora_inicio = datetime.datetime.now()
fecha_inicio = hora_inicio.date()

print()
print("-------------------------------------------")
print("Construcción de la matriz de decisión" )
attributes = ["C1", "C2", "C3", "C4", "C5"]
candidates = ["A1", "A2", "A3", "A4", "A5", "A6","A7","A8","A9"]
n=5
a=9
raw_data = [
    [0.048, 0.047, 0.070, 0.087, 0.190],
    [0.053, 0.052, 0.066, 0.081, 0.058],
    [0.057, 0.057, 0.066, 0.076, 0.022],
    [0.062, 0.062, 0.063, 0.058, 0.007],
    [0.066, 0.066, 0.070, 0.085, 0.004],
    [0.070, 0.071, 0.066, 0.058, 0.003],
    [0.075, 0.075, 0.066, 0.047, 0.002],
    [0.079, 0.079, 0.066, 0.035, 0.002],
    [0.083, 0.083, 0.066, 0.051, 0.000],
]

# Mostrar los datos sin procesar que tenemos
A1 = pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
print(A1)


print("\n -------------------------------------------")
print("Controles iniciales" )
print()
print("Grado de preferencia para cada criterio")
w = [0.400, 0.200, 0.030, 0.070, 0.300]
#w = [0.300, 0.200, 0.200, 0.150, 0.150] #Original del artículo
#w = [0.200, 0.200, 0.200, 0.200, 0.200]
#w = [0.123, 0.099, 0.043, 0.343, 0.392]
weights = pd.Series(w, index=attributes)
print(weights,"\n")

# Los índices de los atributos (de base cero) que se consideran beneficiosos.
# Se supone que los índices no mencionados son atributos de costos.
benefit_attributes = set([0, 1, 2, 3, 4])

############################################################
#---PAso 1 - Normalizando las calificaciones
normalized_matrix = A1.copy()
for i in range(len(attributes)):
    if i in benefit_attributes:
        normalized_matrix.iloc[:, i] = A1.iloc[:, i] / np.linalg.norm(A1.iloc[:, i])
    else:
        normalized_matrix.iloc[:, i] = A1.iloc[:, i] / np.linalg.norm(A1.iloc[:, i], ord=1)

#print("\n-------------------------------------------")
#print("Paso 1: Normalización de la matriz de decisiones")
#print(normalized_matrix)


############################################################
#--- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
#            Multiplicación de la matriz normalizada por los pesos
weighted_matrix = normalized_matrix * weights

#print("\n-------------------------------------------")
#print("Paso 2: Multiplicación de la matriz normalizada por los pesos")
#print(weighted_matrix)


############################################################
#--- Paso 3: Identificar las soluciones ideal y anti-ideal
ideal_best = weighted_matrix.max()
ideal_worst = weighted_matrix.min()

#print("\n-------------------------------------------")
#print("Paso 3: Determinación de las soluciones ideal y anti-ideal")
#print("Ideal:")
#print(ideal_best)
#print("\nAnti-Ideal:")
#print(ideal_worst)


############################################################
#--- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
# Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
s_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
s_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

#print("\n-------------------------------------------")
#print("Paso 4: Cálculo de las distancias a las soluciones ideal y anti-ideal")
#print("Distancias a la solución ideal:")
#print(s_best)
#print("\nDistancias a la solución anti-ideal:")
#print(s_worst)

############################################################
#--- Pasos 5: Cálculo de la puntuación de proximidad relativa
performance_score = s_worst / (s_best + s_worst)

#print("\n-------------------------------------------")
#print("Paso 5: Cálculo de la puntuación de proximidad relativa")
#print("Puntuación de proximidad relativa:")
#print(performance_score)

############################################################
#--- Paso 6: Clasificación de las alternativas
ranked_candidates = performance_score.sort_values(ascending=False)

#print("\n-------------------------------------------")
#print("Paso 6: Clasificación de las alternativas")
#print("Ranking de los candidatos:")
#print(ranked_candidates)



############################################################
Alt = list(ranked_candidates.index)

print("\n-------------------------------------------")
print("La mejor alternativa es:", Alt[0])
print("La clasificación de las alternativas, de manera descendente es: ", Alt )
print()


############################################################
############################################################
# PAra almacenar tiempo de ejecución
hora_fin = datetime.datetime.now()
ejecut=hora_fin-hora_inicio



alternativas = Alt[-5:]
print(alternativas)

# PAra guardar información en archivo de EXCEl
dT= {"Algoritmo": ["TOPSIS"],
    "Hora de inicio": [hora_inicio.time()],
    "Fecha de inicio": [fecha_inicio],
    "Hora de finalización": [hora_fin.time()],
    "Tiempo de ejecución": [ejecut] }

dataT = pd.DataFrame(dT)
dataAlt = pd.DataFrame(Alt)
dataw = pd.DataFrame(w)

with pd.ExcelWriter('Experimentos2/TOPSIS.xlsx', engine='xlsxwriter') as writer:
    #dataI.to_excel(writer, sheet_name='Iniciales')
    dataT.to_excel(writer, sheet_name='Tiempos')
    dataw.to_excel(writer, sheet_name='w')
    A1.to_excel(writer, sheet_name='Matriz')
    dataAlt.to_excel(writer, sheet_name='Ranking_alternativas')

print('Datos guardados el archivo:TOPSIS.xlsx')
print()

# Imprimimos los resultados de tiempo
print("Algoritmo TOPSIS")
print("Hora de inicio:", hora_inicio.time())
print("Fecha de inicio:", fecha_inicio)
print("Hora de finalización:", hora_fin.time())
print("Tiempo de ejecución:",ejecut)
print()
    
    