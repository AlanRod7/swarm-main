# Experimento TOPSIS
#
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# Actualización 24-Feb-2024

import asyncio
import datetime
import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import rankdata


async def ejecutar_topsis(w,n):
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()

    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión" )
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = ["A1", "A2", "A3", "A4", "A5", "A6","A7","A8","A9"]
    #n=6
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
    x = pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    print(x)


    print("\n -------------------------------------------")
    print("Controles iniciales" )
    print()
    print("Grado de preferencia para cada criterio")
    #w = [0.400, 0.200, 0.030, 0.070, 0.300]
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
    normalized_matrix = x.copy()
    for i in range(len(attributes)):
        if i in benefit_attributes:
            normalized_matrix.iloc[:, i] = x.iloc[:, i] / np.linalg.norm(x.iloc[:, i])
        else:
            normalized_matrix.iloc[:, i] = x.iloc[:, i] / np.linalg.norm(x.iloc[:, i], ord=1)

    print("\n-------------------------------------------")
    print("Paso 1: Normalización de la matriz de decisiones")
    print(normalized_matrix)


    ############################################################
    #--- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
    #            Multiplicación de la matriz normalizada por los pesos
    weighted_matrix = normalized_matrix * weights

    print("\n-------------------------------------------")
    print("Paso 2: Multiplicación de la matriz normalizada por los pesos")
    print(weighted_matrix)


    ############################################################
    #--- Paso 3: Identificar las soluciones ideal y anti-ideal
    ideal_best = weighted_matrix.max()
    ideal_worst = weighted_matrix.min()

    print("\n-------------------------------------------")
    print("Paso 3: Determinación de las soluciones ideal y anti-ideal")
    print("Ideal:")
    print(ideal_best)
    print("\nAnti-Ideal:")
    print(ideal_worst)


    ############################################################
    #--- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
    # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
    s_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    print("\n-------------------------------------------")
    print("Paso 4: Cálculo de las distancias a las soluciones ideal y anti-ideal")
    print("Distancias a la solución ideal:")
    print(s_best)
    print("\nDistancias a la solución anti-ideal:")
    print(s_worst)


    ############################################################
    #--- Pasos 5: Cálculo de la puntuación de proximidad relativa
    performance_score = s_worst / (s_best + s_worst)

    print("\n-------------------------------------------")
    print("Paso 5: Cálculo de la puntuación de proximidad relativa")
    print("Puntuación de proximidad relativa:")
    print(performance_score)

    ############################################################
    #--- Paso 6: Clasificación de las alternativas
    ranked_candidates = performance_score.sort_values(ascending=True)

    print("\n-------------------------------------------")
    print("Paso 6: Clasificación de las alternativas")
    print("Ranking de los candidatos:")
    print(ranked_candidates)



    ############################################################
    Alt = list(ranked_candidates.index)

    print("\n-------------------------------------------")
    print("La mejor alternativa es:", Alt[0])
    print("La clasificación de las alternativas, de manera descendente es: ", Alt )
    print()




    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)
    arreglo = ranked_candidates.index[-10:]
    arregloInvertido = tuple((arreglo))
    alternativas = arregloInvertido
    

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

    base_filename = 'Experimentos/TOPSIS'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dT= {"Algoritmo": ["TOPSIS"],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataAlt = pd.DataFrame(Alt)
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
    dataAlt.to_csv(csv_filename, mode='a', index=False)
    print(f'Datos guardados en el archivo CSV: {csv_filename}')
    print()
    
        # Imprimimos los resultados de tiempo
    print("Algoritmo TOPSIS")
    print("Hora de inicio:", hora_inicio.time())
    print("Datos w:", w)
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:",ejecut)
    print()
    
    await asyncio.sleep(0.1)

    datosTopsis = {
        "mejor_alternativa": alternativas,
        "iteraciones": n,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }

    return datosTopsis