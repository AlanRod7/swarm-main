# Experimento MOORA-ACO
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# Actualización 27-Feb-2024
import asyncio
import datetime
import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook


async def ejecutar_mooraaco(EV,w,alpha,beta,rho,Q,n_ants,n_iterations):

    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]
    ress = []

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

    # Contiene las evaluaciones cardinales de cada alternativa respecto a cada criterio
    #EV = ["Min", "Min", "Min", "Min", "Min"]
    print("Evaluaciones cardinales de cada alternativa respecto a cada criterio:")
    print(EV,"\n")

    ### -- Pesos por cada criterio
    #w = [0.400, 0.200, 0.030, 0.070, 0.300]
    #w = [0.300, 0.200, 0.200, 0.150, 0.150] 
    #w = [0.200, 0.200, 0.200, 0.200, 0.200]
    #w = [0.123, 0.099, 0.043, 0.343, 0.392]
    weights = pd.Series(w, index=attributes)
    print("Pesos por cada criterio")
    print(weights)

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
    ResultadoMOORA = []

    ########################################
    ## Calcular la Función Objetivo CON MOORA
    ##########

    ### -- Normalizamos con distancia euclidiana
    # Calcular la suma de los cuadrados de cada fila
    squared_sum = (x ** 2).sum(axis=1)
    #print("squared_sum",squared_sum)

    # Calcular la raíz cuadrada de la suma de cuadrados
    norm_factor = np.sqrt(squared_sum)
    #print("norm_factor",norm_factor)

    # Normalizar dividiendo cada valor por la raíz cuadrada correspondiente
    normalized_data = x.div(norm_factor, axis=0)
    #print("\n Matriz de datos normalizados (distancia euclidiana):")
    #print(normalized_data)

    ### -- Ponderamos la matriz normalizada por los pesos de los criterios
    weighted_data = normalized_data * weights
    #print("\nMatriz de datos ponderados:")
    #print(weighted_data)

    ### -- Cálculo de la puntuación de cada alternativa:
    global_scores = [] ## Crear una lista para almacenar los resultados
    for idx, row in weighted_data.iterrows():
        score = 0  # Inicializamos la puntuación para esta fila
        # Iteramos sobre cada valor en la fila y su correspondiente en 'EV'
        for value, ev_value in zip(row, EV):
            # Sumamos o restamos según el valor en 'EV'
            if ev_value == "Max":
                score += value
            else:
                score -= value
        # Agregamos la puntuación final para esta fila a la lista de resultados
        global_scores.append(score)
    global_scores = pd.Series(global_scores) #Convertir la lista de resultados en una Serie de pandas   
    #print("\nEvaluación de cada alternativa:")
    #print(global_scores)

    ### -- Clasificación de alternativas
    ranked_alternatives = global_scores.sort_values(ascending=False)
    #print("\nClasificación de alternativas:")
    #print(ranked_alternatives)

    ### -- Crear DataFrame para clasificación final
    RankFin = pd.DataFrame(ranked_alternatives, columns=['Puntuación Global'])
    RankFin['Alternativa'] = ranked_alternatives.index
    RankFin.reset_index(drop=True, inplace=True)

    print("\nClasificación Final:")
    print(RankFin)

    print("\nLa mejor solución es la alternativa:", RankFin.iloc[0]['Alternativa'])

    ##########################################################################
    # Agregar los valores de Alternativa a la lista ResultadoMOORA
    ResultadoMOORA.extend(RankFin['Alternativa'])

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
    print("Resultado de MOORA:", ' '.join(['A'+str(i) for i in ResultadoMOORA]))
    print()
    print("Resultados de MOORA-ACO:") 
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
    print("Algoritmo MOORA-ACO")
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:", ejecut)
    print("  ---------------------------------")
    print()



    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/MOORAACO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dI = {"alpha": [alpha], "beta": [beta], "Tasa de evaporación de feromona(rho)": [rho], "Cantidad de feromona depositada(Q)": [Q], "Número de hormigas(n_ants)":[n_ants], "No_ejecuciones del programa":[itera_max]}
    dT= {"Algoritmo": ["MOORA-ACO"],
        "Cantidad de repeticiones del programa": [n_iterations],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataI = pd.DataFrame(dI)
    dataOrig=pd.DataFrame(raw_data)
    dataResult = pd.DataFrame(resultados)
    dataMOO=pd.DataFrame(ResultadoMOORA)
    dataw = pd.DataFrame(w)
    dataND = pd.DataFrame(normalized_data)
    datawd= pd.DataFrame(weighted_data)
    datags= pd.DataFrame(global_scores)
    dataECA = pd.DataFrame(EV)
    dataAlt = pd.DataFrame(RankFin)

    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos', index=False)
        dataMOO.to_excel(writer, sheet_name='Resultados_MOORA')
        dataResult.to_excel(writer, sheet_name='Resultados_iteración')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataI.to_excel(writer, sheet_name='Variables_Iniciales')
        dataw.to_excel(writer, sheet_name='w')
        datags.to_excel(writer, sheet_name='Evaluación_cada_alternativa')
        dataECA.to_excel(writer, sheet_name='Ev_cardinales_alternativa')
        dataND.to_excel(writer, sheet_name='Matriz_normaliza')
        datawd.to_excel(writer, sheet_name='Matriz_ponderada')
        dataAlt.to_excel(writer, sheet_name='Ranking_alternativas')

        # Ajustar automáticamente el ancho de las columnas en la hoja 'Tiempos'
        worksheet = writer.sheets['Tiempos']
        for i, col in enumerate(dataT.columns):
            column_len = max(dataT[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_len)
    print(f'Datos guardados en el archivo: {excel_filename}')

    ### -- Guardar los mismos datos en un archivo CSV con el mismo número
    csv_filename = f'{base_filename}_{counter}.csv'
    dataT.to_csv(csv_filename, index=False)
    dataMOO.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataI.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    datags.to_csv(csv_filename, mode='a', index=False)
    dataECA.to_csv(csv_filename, mode='a', index=False)
    dataND.to_csv(csv_filename, mode='a', index=False)
    datawd.to_csv(csv_filename, mode='a', index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)

    print(f'Datos guardados en el archivo: {csv_filename}')
    print()




    await asyncio.sleep(0.1)

    datosMooraaco = {
        "mejor_alternativa": numeros,
        "iteraciones": n_iterations,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }

    return datosMooraaco