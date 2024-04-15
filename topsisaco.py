# Experimento TOPSIS-ACO
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# Actualización 27-Feb-2024

import asyncio
import datetime
import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook


async def ejecutar_topsisaco(w,alpha,beta,rho,Q,n_ants,n_iterations): # Elimine benefit_attributes por que no estoy seguro de su implementacion

    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]
    ress=[]

    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión")
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    n = 5
    a = 9
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
    x = pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    print(x)

    print("\n-------------------------------------------")
    print("Controles iniciales")

    ### -- Pesos por cada criterio
    #w = [0.400, 0.200, 0.030, 0.070, 0.300]
    #w = [0.300, 0.200, 0.200, 0.150, 0.150] 
    #w = [0.200, 0.200, 0.200, 0.200, 0.200]
    #w = [0.123, 0.099, 0.043, 0.343, 0.392]
    weights = pd.Series(w, index=attributes)
    print("Pesos por cada criterio")
    print(weights)

    # Los índices de los atributos (de base cero) que se consideran beneficiosos.
    # Se supone que los índices no mencionados son atributos de costos.
    benefit_attributes = set([0, 1, 2, 3, 4])

    #alpha = 1     # Peso de la feromona
    #beta = 2      # Peso de la heurística
    #rho = 0.1     # Tasa de evaporación de feromona
    #Q = 100       # Cantidad de feromona depositada
    #n_ants = 10   # Número de hormigas
    #n_iterations = 100  # Número de iteraciones

    itera_max = 10      # Número de ejecuciones de ACO

    # DataFrame para almacenar los resultados
    resultados = pd.DataFrame(columns=['Ejecución:   ','  Mejor_Alternativa'])

    # Lista para almacenar los resultados de MOORA
    ResultadoTOPSIS = []

    ########################################
    ## Calcular la Función Objetivo CON TOPSIS
    ##########

    #---PAso 1 - Normalizando las calificaciones
    normalized_matrix = x.copy()
    for i in range(len(attributes)):
        if i in benefit_attributes:
            normalized_matrix.iloc[:, i] = x.iloc[:, i] / np.linalg.norm(x.iloc[:, i])
        else:
            normalized_matrix.iloc[:, i] = x.iloc[:, i] / np.linalg.norm(x.iloc[:, i], ord=1)
    #print("Paso 1: Normalización de la matriz de decisiones")
    #print(normalized_matrix)

    #--- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
    weighted_matrix = normalized_matrix * weights
    #print("Paso 2: Multiplicación de la matriz normalizada por los pesos")
    #print(weighted_matrix)

    #--- Paso 3: Identificar las soluciones ideal y anti-ideal
    ideal_best = weighted_matrix.max()
    ideal_worst = weighted_matrix.min()
    #print("Paso 3: Determinación de las soluciones ideal y anti-ideal")
    #print("Ideal:")
    #print(ideal_best)
    #print("\nAnti-Ideal:")
    #print(ideal_worst)

    #--- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
    s_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    #print("Paso 4: Cálculo de las distancias a las soluciones ideal y anti-ideal")
    #print("Distancias a la solución ideal:")
    #print(s_best)
    #print("\nDistancias a la solución anti-ideal:")
    #print(s_worst)

    #--- Pasos 5: Cálculo de la puntuación de proximidad relativa
    performance_score = s_worst / (s_best + s_worst)
    #print("Paso 5: Cálculo de la puntuación de proximidad relativa")
    #print("Puntuación de proximidad relativa:")
    #print(performance_score)

    #--- Paso 6: Clasificación de las alternativas
    ranked_candidates = performance_score.sort_values(ascending=True)
    #print("Paso 6: Clasificación de las alternativas")
    #print("Ranking de los candidatos:")
    #print(ranked_candidates)


    ### -- Crear DataFrame para clasificación final
    RankFin = pd.DataFrame(ranked_candidates, columns=['Puntuación Global'])
    RankFin['Alternativa'] = ranked_candidates.index
    RankFin.reset_index(drop=True, inplace=True)

    print("\nClasificación Final:")
    print(RankFin)

    print("\nLa mejor solución es la alternativa:", RankFin.iloc[0]['Alternativa'])

    ##########################################################################
    # Agregar los valores de Alternativa a la lista ResultadoMOORA
    ResultadoTOPSIS.extend(RankFin['Alternativa'])

    # Obteniendo la mejor puntuación global para inicializar la feromona en ACO
    best_score = RankFin.iloc[0]['Puntuación Global']

    ##########################################################################
    # ACO
    for iteracion_total in range(itera_max):

        # Inicialización de feromonas con la mejor puntuación global de MOORA
        pheromone = np.ones((len(attributes), len(candidates))) * best_score  
        
        # Función para calcular la probabilidad de selección de una alternativa dado un atributo
        def calculate_probabilities(attribute, pheromone, heuristic):
            probabilities = (pheromone[attribute] ** alpha) * (heuristic ** beta)
            # Manejar divisiones por cero y valores infinitos
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=np.nanmax(probabilities))
            # Normalizar las probabilidades
            total_prob = np.sum(probabilities)
            if total_prob == 0:  # Si todas las probabilidades son cero, establecerlas uniformemente
                probabilities = np.ones_like(probabilities) / len(candidates)
            else:
                probabilities /= total_prob
            return probabilities

        # Ciclo principal del algoritmo ACO
        for iteration in range(n_iterations):
            for ant in range(n_ants):
                heuristic = 1 / (x.values + 1e-10)  # Heurística simple inversa de la matriz de datos con pequeña constante
                selected_alternatives = []
                
                for attribute in range(len(attributes)):
                    probabilities = calculate_probabilities(attribute, pheromone, heuristic[:, attribute])
                    # Asegurar que las probabilidades sean no negativas
                    probabilities = np.maximum(probabilities, 0)
                    # Re-normalizar las probabilidades
                    probabilities /= np.sum(probabilities)
                    selected_alternative = np.random.choice(len(candidates), p=probabilities)
                    selected_alternatives.append(selected_alternative)
                
                # Actualizar feromonas
                for attribute, selected_alternative in enumerate(selected_alternatives):
                    pheromone[attribute, selected_alternative] += Q / (x.values[selected_alternative, attribute] + 1e-10)
            
            # Evaporación de feromonas
            pheromone *= (1 - rho)

        # Determinar la mejor alternativa
        best_alternative_index = np.argmax(np.sum(x.values.T * pheromone, axis=1))
        best_alternative = candidates[best_alternative_index]
        ress.append(best_alternative)

        resultados = pd.concat([resultados, pd.DataFrame({'Ejecución:   ': [iteracion_total+1], '  Mejor_Alternativa': [best_alternative]})], ignore_index=True)

    # Imprimir resultados
    print()
    print("--------------------------------------------------------------------------------------------")
    print("Resultado de TOPSIS:", ResultadoTOPSIS)
    print()
    print("Resultados de TOPSIS-ACO:") 
    print(resultados)

    numeros = [int(caracter) for elemento in ress for caracter in elemento if caracter.isdigit()]
    numeros=numeros[-10:]


    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)

    # Imprimimos los resultados de tiempo
    print()
    print("Algoritmo TOPSIS")
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:",ejecut)
    print()


    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/TOPSISACO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dT= {"Algoritmo": ["TOPSIS-ACO"],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataAlt = pd.DataFrame(ResultadoTOPSIS)
    dataw = pd.DataFrame(w)
    dataOrig=pd.DataFrame(raw_data)
    dataBen=pd.DataFrame(benefit_attributes)
    dataNMx=pd.DataFrame(normalized_matrix)
    dataNMxW=pd.DataFrame(weighted_matrix)
    dataIb=pd.DataFrame(ideal_best)
    dataIw=pd.DataFrame(ideal_worst)
    dataSBt=pd.DataFrame(s_best)
    dataSwT=pd.DataFrame(s_worst)
    datapsc=pd.DataFrame(performance_score)

        
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos')
        dataResult = pd.DataFrame(resultados)
        dataAlt.to_excel(writer, sheet_name='Resultados TOPSIS')
        dataResult.to_excel(writer, sheet_name='Resultados_ejecución')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataw.to_excel(writer, sheet_name='w')
        dataBen.to_excel(writer, sheet_name='Atributos_beneficios')
        dataNMx.to_excel(writer, sheet_name='Matriz_normalizada')
        dataNMxW.to_excel(writer, sheet_name='Matriz_normalizada_xPesos')
        dataIb.to_excel(writer, sheet_name='Solución_ideal(SI)')
        dataIw.to_excel(writer, sheet_name='Solución_anti-ideal(SaI)')
        dataSBt.to_excel(writer, sheet_name='Distancia_SI')
        dataSwT.to_excel(writer, sheet_name='Distancia_SaI')
        datapsc.to_excel(writer, sheet_name='Puntuación_prox_relativa')
        

        # Ajustar automáticamente el ancho de las columnas en la hoja 'Tiempos'
        worksheet = writer.sheets['Tiempos']
        for i, col in enumerate(dataT.columns):
            column_len = max(dataT[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_len)
    print(f'Datos guardados en el archivo: {excel_filename}')


    ### -- Guardar los mismos datos en un archivo CSV con el mismo número
    csv_filename = f'{base_filename}_{counter}.csv'
    dataT.to_csv(csv_filename, index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    dataBen.to_csv(csv_filename, mode='a', index=False)
    dataNMx.to_csv(csv_filename, mode='a', index=False)
    dataNMxW.to_csv(csv_filename, mode='a', index=False)
    dataIb.to_csv(csv_filename, mode='a', index=False)
    dataIw.to_csv(csv_filename, mode='a', index=False)
    dataSBt.to_csv(csv_filename, mode='a', index=False)
    dataSwT.to_csv(csv_filename, mode='a', index=False)
    datapsc.to_csv(csv_filename, mode='a', index=False)

    print(f'Datos guardados en el archivo: {csv_filename}')
    print()

    await asyncio.sleep(0.1)

    datosTopsisaco = {
        "mejor_alternativa": numeros,
        "iteraciones": n_iterations,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }

    return datosTopsisaco