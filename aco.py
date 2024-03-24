# Experimento ACO
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# ACtualizado 27-Feb-2024

import numpy as np
import pandas as pd
import datetime
from openpyxl import load_workbook
import os
import asyncio

async def ejecutar_aco(w,alpha,gamma,iter_max):

    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]

    # Datos de entrada
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    datarw = {
        'C1': [0.048, 0.053, 0.057, 0.062, 0.066, 0.070, 0.075, 0.079, 0.083],
        'C2': [0.047, 0.052, 0.057, 0.062, 0.066, 0.071, 0.075, 0.079, 0.083],
        'C3': [0.070, 0.066, 0.066, 0.063, 0.070, 0.066, 0.066, 0.066, 0.066],
        'C4': [0.087, 0.081, 0.076, 0.058, 0.085, 0.058, 0.047, 0.035, 0.051],
        'C5': [0.190, 0.058, 0.022, 0.007, 0.004, 0.003, 0.002, 0.002, 0.000]
    }
    xP = pd.DataFrame(data=datarw, index=candidates)

    # Parámetros del algoritmo ACO
    alpha = alpha     # Peso de la feromona [0.1, 10]-común(1): Controla la influencia de la feromona en la elección de las hormigas. Valores más altos dan más peso a la feromona.
    beta = gamma      # Peso de la heurística [1,5]-común(2): Controla la influencia de la información heurística en la elección de las hormigas. Valores más altos dan más peso a la heurística.
    rho = 0.1     # Tasa de evaporación de feromona [0.1, 0.5]-común(0.1): Controla la tasa a la que la feromona se evapora en cada iteración. Valores más altos indican una tasa de evaporación más rápida.
    Q = 100       # Cantidad de feromona depositada [10,1000]-común(100): Especifica la cantidad de feromona depositada por cada hormiga en cada iteración. Valores más altos aumentan la cantidad de feromona depositada.
    n_ants = 10    # Número de hormigas [10,100]-común(10-50): Especifica el número de hormigas que se utilizan en cada iteración del algoritmo. Valores más altos pueden aumentar la diversidad y la capacidad de búsqueda, pero también aumentan el costo computacional.
    n_iterations = iter_max  # Número de iteraciones [10,100]-común(100-500): Especifica el número de iteraciones que realiza el algoritmo ACO. Un mayor número de iteraciones permite una búsqueda más exhaustiva, pero también aumenta el tiempo de ejecución.

    #itera_max = 10 # Número de ejecuciones de ACO

    # DataFrame para almacenar los resultados
    resultados = pd.DataFrame(columns=['Ejecución:   ','  Mejor_Alternativa'])
   
    #for iteracion_total in range(itera_max):


        # Inicialización de feromonas
    pheromone = np.ones((len(attributes), len(candidates)))

    # Función para calcular la probabilidad de selección de una alternativa dado un atributo
    def calculate_probabilities(attribute, pheromone, heuristic):
        probabilities = (pheromone[attribute] ** alpha) * (heuristic ** beta)
        # Asegurar que las probabilidades sean no negativas
        probabilities[probabilities < 0] = 0
        # Normalizar las probabilidades
        probabilities /= np.sum(probabilities)
        return probabilities

    # Ciclo principal del algoritmo ACO
    for iteration in range(n_iterations):
        for ant in range(n_ants):
            heuristic = 1 / (xP.values + 1e-10)  # Heurística simple inversa de la matriz de datos con pequeña constante
            selected_alternatives = []
            
            for attribute in range(len(attributes)):
                probabilities = calculate_probabilities(attribute, pheromone, heuristic[:, attribute])
                selected_alternative = np.random.choice(len(candidates), p=probabilities)
                selected_alternatives.append(selected_alternative)
            
            # Actualizar feromonas
            for attribute, selected_alternative in enumerate(selected_alternatives):
                pheromone[attribute, selected_alternative] += Q / (xP.values[selected_alternative, attribute] + 1e-10)
        
        # Evaporación de feromonas
        pheromone *= (1 - rho)

        # Determinar la mejor alternativa
        best_alternative_index = np.argmax(np.sum(xP.values.T * pheromone, axis=1))
        best_alternative = candidates[best_alternative_index]
       
        
        #print("La mejor alternativa es:", best_alternative)
        
        resultados = pd.concat([resultados, pd.DataFrame({'Ejecución:   ': [iter_max+1], '  Mejor_Alternativa': [best_alternative]})], ignore_index=True)
        


    # Imprimir resultados
    print()
    print("--------------------------------------------------------------------------------------------")
    print("Resultados de ACO:") 
    print(resultados)

    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)
    #arreglo = best_alternative[candidates[-10:]]
    
    #alternativas = tuple((arreglo))
    
   
    
    
    #alternativas = resultados[best_alternative][-10:]
   


    

    # Imprimimos los resultados de tiempo
    print()
    print("Algoritmo ACO")
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:", ejecut)
    print("  ---------------------------------")
    print()



    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/ACO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dI = {"alpha": [alpha], "beta": [beta], "Tasa de evaporación de feromona(rho)": [rho], "Cantidad de feromona depositada(Q)": [Q], "Número de hormigas(n_ants)":[n_ants], "No_ejecuciones del programa":[iter_max]}
    dT= {"Algoritmo": ["ACO"],
        "Cantidad de repeticiones del programa": [n_iterations],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataI = pd.DataFrame(dI)
    dataOrig=pd.DataFrame(datarw)
    dataResult = pd.DataFrame(resultados)
    
    #dataMOO=pd.DataFrame(ResultadosMOORA)

    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos', index=False)
        #dataMOO.to_excel(writer, sheet_name='Resultados_MOORA')
        dataResult.to_excel(writer, sheet_name='Resultados_ejecución')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataI.to_excel(writer, sheet_name='Variables_Iniciales')

        # Ajustar automáticamente el ancho de las columnas en la hoja 'Tiempos'
        worksheet = writer.sheets['Tiempos']
        for i, col in enumerate(dataT.columns):
            column_len = max(dataT[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_len)
    print(f'Datos guardados en el archivo: {excel_filename}')

    ### -- Guardar los mismos datos en un archivo CSV con el mismo número
    csv_filename = f'{base_filename}_{counter}.csv'
    dataT.to_csv(csv_filename, index=False)
    #dataMOO.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataI.to_csv(csv_filename, mode='a', index=False)
    print(f'Datos guardados en el archivo: {csv_filename}')
    print()

    await asyncio.sleep(0.1)
    #alternativas = [int(value) for value in alternativas]

    datosAco = {
        "mejor_alternativa": 3,
        "iteraciones": n_iterations,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }

    return datosAco