
import asyncio
from ipaddress import v4_int_to_packed
from flask import Flask, render_template, request
from openpyxl import load_workbook
import os
import random
from decimal import Decimal
import pandas as pd
import numpy as np
import xlsxwriter
from cProfile import label
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import datetime


async def ejecutar_da(w):

    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()

    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión" )
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    n=5 #Criterios
    a=9 # Alternativas
    Resultados=[]

    datarw={'C1':[0.048,0.053,0.057,0.062,0.066,0.070,0.075,0.079,0.083],
        'C2':[0.047,0.052,0.057,0.062,0.066,0.071,0.075,0.079,0.083],
        'C3':[0.070,0.066,0.066,0.063,0.070,0.066,0.066,0.066,0.066],
        'C4':[0.087,0.081,0.076,0.058,0.085,0.058,0.047,0.035,0.051],
        'C5':[0.190,0.058,0.022,0.007,0.004,0.003,0.002,0.002,0.000]}
    xP=pd.DataFrame(data=datarw,index=candidates)
    print(xP,"\n")

    #############################################################################################
    print("\n -------------------------------------------")
    print("Controles iniciales" )

    print("-------------------------------------------")
    print("Grado de preferencia para cada alternativa")
    ### -- Pesos por cada criterio
    #w=[float(0.200),float(0.200),float(0.200),float(0.200),float(0.200)]
    #w=[float(0.400),float(0.200),float(0.030),float(0.070),float(0.300)]
    #w=[float(0.123),float(0.099),float(0.043),float(0.343),float(0.392)]
    print(w,"\n")



    ### -- Solución ideal
    St=[]
    print("-------------------------------------------")
    print("Paso 1: Establecer la solución ideal")
    for j in range(n):
        P1=0
        for i in range(a):
            P1 += round(xP.iat[i,j],3)
        P2=round((float(P1)/float(a)),3)
        St.append(round((float(P2)),3))
    S= pd.DataFrame(St, columns=["    Solución ideal"])
    print(S,"\n")


    ### -- Índice de similitud
    print("-------------------------------------------")
    print("PAso 2: Determinar el índice de similitud")
    #a) normalizamos (a/S)
    #b) elevar lo normalizado al peso(w)
    #c) Producto sucesivo
    CFt=[]
    SI1=[]
    PST=[]

    ISSFO = pd.DataFrame(columns=attributes)
    for j in range(a):
        SI1=[]
        ISSFOm1=[]
        for i in range(n):
            dat1= float(xP.iat[j,i])
            dat2 = float(S.iat[i, 0]) 
            dat3 = round((dat1/dat2),3)

            wn2=float(w[i])
            dat4 = round((abs(dat3)**abs(wn2)),3)
            ISSFOm1.append(round((float(dat4)),3))

        ISSFOVr1 = pd.DataFrame({'C1':[ISSFOm1[0]],'C2':[ISSFOm1[1]],'C3':[ISSFOm1[2]],'C4':[ISSFOm1[3]],'C5':[ISSFOm1[4]]})
        ISSFO = pd.concat([ISSFO,ISSFOVr1], ignore_index=True)
    #print("ISSFO(1)=")
    #print(ISSFO,"\n")

    for j in range(a):
        Sqq1=float(1)   
        for z in range(n):
            dat5= float(ISSFO.iat[j,z])
            Sqq1=(Sqq1*dat5)

        Sqq1=round(Sqq1,3)
        CFt.append(float(Sqq1))
        PST.append(float(Sqq1))
    PSS = pd.DataFrame(PST, columns=["    Índice de similitud"]) 
    #print("-- Índice de similitud =")
    print(PSS)


    ### -- Clasificación de alternativas
    print("\n -------------------------------------------")
    print("PAso 3: Establecer el ranking de las alternativas, en orden descendente.")
    PST = pd.DataFrame(PST, columns=["Índice de similitud"])
    PST.sort_values(by="Índice de similitud", ascending=True, inplace=True) # Ordenar el DataFrame PST por el índice de similitud del menor al mayor
    #print("   Producto sucesivo=   ")
    print(PST)


    #####################################################################################
    ### -- Crear DataFrame para clasificación final
    RankFin = pd.DataFrame({
        'Puntuación Global': PST['Índice de similitud'].values,
        'Alternativa': PST.index})

    print("\n -------------------------------------------")
    print("\nClasificación Final:")
    print(RankFin)


    print(" -------------------------------------------")
    print("\n                                La mejor solución es la alternativa:", RankFin.iloc[0]['Alternativa'], "con una puntuación global de:", RankFin.iloc[0]['Puntuación Global'])



    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)

    # Imprimimos los resultados de tiempo
    print("Método Análisis Dimensional(DA)")
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:",hora_fin-hora_inicio)
    print("Tiempo de ejecución:", ejecut)
    print()
    arreglo = RankFin.index[::-1]
    arregloInvertido = tuple((arreglo))
    alternativas = arregloInvertido
    

    ####################################################################################
    ### Para guardar información en archivo de EXCEL

    base_filename = 'Experimentos/DA'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dT = {"Método": ["DA"],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [ejecut]}

    dataT = pd.DataFrame(dT)
    dataAlt = pd.DataFrame(RankFin)
    dataw = pd.DataFrame(w)
    dataSI = pd.DataFrame(S)
    dataDIS= pd.DataFrame(PSS)
    dataPS= pd.DataFrame(PST)


    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos')
        dataAlt.to_excel(writer, sheet_name='Ranking_alternativas')
        xP.to_excel(writer, sheet_name='Matriz_decisión')
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
    dataT.to_csv(csv_filename, mode='a', index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)
    xP.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    dataSI.to_csv(csv_filename, mode='a', index=False)
    dataDIS.to_csv(csv_filename, mode='a', index=False)
    dataPS.to_csv(csv_filename, mode='a', index=False)
    print(f'Datos guardados en el archivo CSV: {csv_filename}')
    print()

    await asyncio.sleep(0.1)

    datosDa = {
        "mejor_alternativa": alternativas,
        "iteraciones": 10,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }

    return datosDa