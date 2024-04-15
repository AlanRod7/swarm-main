
import asyncio
import datetime
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook


async def ejecutar_daaco(w, alpha, beta, rho, Q, n_ants, iter_max):

    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]
    ress = []
    best_ress = []

    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión")
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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

    #alpha = 1     # Peso de la feromona
    #beta = 2      # Peso de la heurística
    #rho = 0.1     # Tasa de evaporación de feromona
    #Q = 100       # Cantidad de feromona depositada
    #n_ants = 10   # Número de hormigas
    #n_iterations = 100  # Número de iteraciones

    #itera_max = n_iterations      # Número de ejecuciones de ACO

    # DataFrame para almacenar los resultados
    resultados = pd.DataFrame(columns=['Ejecución:   ','  Mejor_Alternativa'])

    # Lista para almacenar los resultados de MOORA
    ResultadoDA = []

    ########################################
    ## Calcular la Función Objetivo CON TOPSIS
    ##########

    #---PAso 1 - Normalizando las calificaciones
    ### -- Solución ideal
    St=[]
    for j in range(n):
        P1=0
        for i in range(a):
            P1 += round(x.iat[i,j],3)
        P2=round((float(P1)/float(a)),3)
        St.append(round((float(P2)),3))
    S= pd.DataFrame(St, columns=["    Solución ideal"])
    #print(S,"\n")

    ### -- Índice de similitud
    CFt=[]
    SI1=[]
    PST=[]
    ISSFO = pd.DataFrame(columns=attributes)
    for j in range(a):
        SI1=[]
        ISSFOm1=[]
        for i in range(n):
            dat1= float(x.iat[j,i])
            dat2 = float(S.iat[i, 0]) 
            dat3 = round((dat1/dat2),3)
            wn2=float(w[i])
            dat4 = round((abs(dat3)**abs(wn2)),3)
            ISSFOm1.append(round((float(dat4)),3))
        ISSFOVr1 = pd.DataFrame({'C1':[ISSFOm1[0]],'C2':[ISSFOm1[1]],'C3':[ISSFOm1[2]],'C4':[ISSFOm1[3]],'C5':[ISSFOm1[4]]})
        ISSFO = pd.concat([ISSFO,ISSFOVr1], ignore_index=True)

    for j in range(a):
        Sqq1=float(1)   
        for z in range(n):
            dat5= float(ISSFO.iat[j,z])
            Sqq1=(Sqq1*dat5)
        Sqq1=round(Sqq1,3)
        CFt.append(float(Sqq1))
        PST.append(float(Sqq1))
    PSS = pd.DataFrame(PST, columns=["    Índice de similitud"]) 
    #print(PSS)

    ### -- Clasificación de alternativas
    PST = pd.DataFrame(PST, columns=["Índice de similitud"])
    PST.sort_values(by="Índice de similitud", ascending=True, inplace=True) # Ordenar el DataFrame PST por el índice de similitud del menor al mayor
    #print("   Producto sucesivo=   ")
    #print(PST)


    ### -- Crear DataFrame para clasificación final
    RankFin = pd.DataFrame({
        'Puntuación Global': PST['Índice de similitud'].values,
        'Alternativa': PST.index})

    print("\n -------------------------------------------")
    print("\nClasificación Final:")
    print(RankFin)

    print("\nLa mejor solución es la alternativa:", RankFin.iloc[0]['Alternativa'])

    ##########################################################################
    # Agregar los valores de Alternativa a la lista ResultadoMOORA
    ResultadoDA.extend(RankFin['Alternativa'])

    # Obteniendo la mejor puntuación global para inicializar la feromona en ACO
    best_score = RankFin.iloc[0]['Puntuación Global']

    ##########################################################################
    # ACO
    for iteracion_total in range(iter_max):

        # Inicialización de feromonas con la mejor puntuación global de DA
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
        for iteration in range(iter_max):
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
        #best_ress.appendbest_alternative

        resultados = pd.concat([resultados, pd.DataFrame({'Ejecución:   ': [iteracion_total+1], '  Mejor_Alternativa': [best_alternative]})], ignore_index=True)

    # Imprimir resultados
    print()
    print("--------------------------------------------------------------------------------------------")
    print("Resultado de DA:", ResultadoDA)
    print()
    print("Resultados de DA-ACO:") 
    print(resultados)

    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)

    # Imprimimos los resultados de tiempo
    print()
    print("Algoritmo DA-ACO")
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:",ejecut)
    print()


    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/DAACO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dT= {"Algoritmo": ["DA-ACO"],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataRK = pd.DataFrame(ResultadoDA)
    dataOrig=pd.DataFrame(raw_data)
    dataAlt = pd.DataFrame(RankFin)
    dataw = pd.DataFrame(w)
    dataSI = pd.DataFrame(S)
    dataDIS= pd.DataFrame(PSS)
    dataPS= pd.DataFrame(PST)

        
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos')
        dataResult = pd.DataFrame(resultados)
        dataRK.to_excel(writer, sheet_name='Resultados DA')
        dataResult.to_excel(writer, sheet_name='Resultados_ejecución')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataw.to_excel(writer, sheet_name='w')
        dataSI.to_excel(writer, sheet_name='Solución_ideal')
        dataDIS.to_excel(writer, sheet_name='Índice_similitud')
        dataPS.to_excel(writer, sheet_name='Produto_sucesivo')
    

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
    dataRK.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    dataSI.to_csv(csv_filename, mode='a', index=False)
    dataDIS.to_csv(csv_filename, mode='a', index=False)
    dataPS.to_csv(csv_filename, mode='a', index=False)


    print(f'Datos guardados en el archivo: {csv_filename}')
    print()

    await asyncio.sleep(0.1)

    datosDaaco = {
        "mejor_alternativa": ress[-10:],
        "iteraciones": iter_max,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }

    return datosDaaco