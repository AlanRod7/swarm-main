# Experimento MOORA-PSO
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# ACtualizado 06/Feb/2023

import asyncio
import datetime
import math
import os
import random
from cProfile import label
from decimal import Decimal
from ipaddress import v4_int_to_packed
from re import X

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter
from flask import Flask, render_template, request
from matplotlib.transforms import Bbox
from openpyxl import load_workbook


async def ejecutar_moorapso(w, wwi, c1, c2, T, r1, r2): 
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]
    ResultadosMoora=pd.DataFrame()
    Comparativo=pd.DataFrame()

    print("-------------------------------------------")
    print("Construcción de la matriz de decisión" )
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    n=5 # criterios
    a=9 # alternativas

    A1t={'C1':[0.048,0.053, 0.057, 0.062, 0.066, 0.070, 0.075, 0.079, 0.083],
        'C2':[0.047, 0.052, 0.057, 0.062, 0.066, 0.071, 0.075, 0.079, 0.083],
        'C3':[0.070, 0.066, 0.066, 0.063, 0.070, 0.066, 0.066, 0.066, 0.066],
        'C4':[0.087, 0.081, 0.076, 0.058, 0.085, 0.058, 0.047, 0.035, 0.051],
        'C5':[0.190, 0.058, 0.022, 0.007, 0.004, 0.003, 0.002, 0.002, 0.000] }
    x=pd.DataFrame(A1t)
    print(x,"\n")

    ### -- Contien las evaluaciones cardinales de cada alternativa respecto a cada criterio
    EV=["Min", "Min", "Min","Min", "Min"]
    #EV=pd.DataFrame(Ev1) # Contien las evaluaciones cardinales de cada alternativa respecto a cada criterio
    #print(EV,"\n")

    #################################################################
    ### --- Controles iniciales \n" 
    #wwi=0.7 # Tener un rango menor ayuda a
    #c1=2.5    # Este influye en la pobabilidad hacia
    #c2=2.5    # Este influye en la pobabilidad hacia
    rangoMin=0 #este rango de valores
    rangoMax=1  
    #T=10    #número de iteraciones para PSO
    dim=n*a #dimensión del enjambre

    #Pesos por cada criterio
    #w=[float(0.123),float(0.099),float(0.043),float(0.343),float(0.392)]
    #w=[float(0.2),float(0.2),float(0.2),float(0.2),float(0.2)]
    #w=[float(0.400),float(0.200),float(0.030),float(0.070),float(0.300)]
    #print("Pesos por criterio",w)



    #####################
    # **********************************************************************PRIMERA ITERACIÓN (inicial)
    print("ITERACIÓN # 1 -----------------------")

    ####################
    ## MOORA

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
    weighted_data = normalized_data * w
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
    #print("\nClasificación Final:")
    #print(RankFin)


    ####################
    ## PSO

    ### -- Asignamos las primeras posiciones de MOOORA como r1 y r2
    Valor1 = int(RankFin.iat[0,1]) # Esta es la primera Mejor alternativa
    Valor2 = int(RankFin.iat[1,1]) # esta es la segunda mejor alternativa
    #print("Valores", Valor1, Valor2)

    # -- Actualizar los valores para r1 y r2
    r1 = pd.DataFrame(columns=attributes)
    r2 = pd.DataFrame(columns=attributes)
    r1=x.iloc[Valor1]
    r2=x.iloc[Valor2]


    ### -- Dato para comparativos
    ValorFin = []
    ValorFin.append(int(RankFin.iat[0,1]))
    ValorFin.append(0)
    CMp1 = pd.DataFrame({'MOORA':[ValorFin[0]],'MOORAFIN':[ValorFin[1]]})
    Comparativo = pd.concat([Comparativo,CMp1], ignore_index=True)
    #print("Comparativo")
    #print(Comparativo)


    print("------------------------------------------- \n")
    print("Controles iniciales:")
    print("w(inertia) = ",wwi)
    print("c1 = ",c1)
    print("c2 = ",c2)
    print("No. de iteraciones = ",T,"\n")
    print("r1 = \n", r1,"\n")
    print("r2 = \n", r2,"\n")
    print("Rango de valores: (",rangoMin,",",rangoMax,")")
    print("------------------------------------------- \n")


    ### -- CURRENT VELOCITY (V)
    V = pd.DataFrame(columns=attributes)
    for _ in range(a):
        Vram1 = [round(random.uniform(rangoMin, rangoMax), 3) for _ in range(n)]
        Ver1 = pd.DataFrame([Vram1], columns=attributes)
        V = pd.concat([V, Ver1], ignore_index=True)
    print("V(1)=")
    print(V,"\n")



    ### -- CURRENT POSITION (CP)
    #LA PRIMERA MEJOR POSICIÓN, SIEMPRE SERA LA PRIMERA POSICIÓN, NO SE TIENE ANTEDECENTES
    CP=pd.DataFrame(A1t) # Esta es la primera posición del enjambre, que corresponde a la matriz original
    print("CP(1)=")
    print(CP,"\n")


    #############################
    # FUNCIÓN OBJETIVO, CURRENT FITNESS (CF = Fx)
    #print("   Evaluar la función objetivo para obtener el mejor local y global.")

    ### -- Normalizamos con distancia euclidiana
    # Calcular la suma de los cuadrados de cada fila
    squared_sum = (CP ** 2).sum(axis=1)
    #print("squared_sum",squared_sum)

    # Calcular la raíz cuadrada de la suma de cuadrados
    norm_factor = np.sqrt(squared_sum)
    #print("norm_factor",norm_factor)

    # Normalizar dividiendo cada valor por la raíz cuadrada correspondiente
    normalized_data = CP.div(norm_factor, axis=0)
    #print("\n Matriz de datos normalizados (distancia euclidiana):")
    #print(normalized_data)

    ### -- Ponderamos la matriz normalizada por los pesos de los criterios
    weighted_data = normalized_data * w
    #print("\nMatriz de datos ponderados:")
    #print(weighted_data)


    ### -- Cálculo de la puntuación de cada alternativa:
    CF = pd.Series(dtype=float)
    Fx= pd.Series(dtype=float)
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

        #Este valor también corresponde a la PRIMERA mejor posicion local (CF) y Funcion objetivo(Fx)
        CFxe0 = pd.Series(score)
        CF = pd.concat([CF,CFxe0], ignore_index=True)

        Fx120 = pd.Series(score)
        Fx = pd.concat([Fx,Fx120], ignore_index=True)

    global_scores = pd.Series(global_scores) #Convertir la lista de resultados en una Serie de pandas
    #print("CF NUEVO=")
    #print(CF,"\n")
    #print("LBF=")
    #print(Fx,"\n")
    #print("\nEvaluación de cada alternativa:")
    #print(global_scores)


    ### -- Clasificación de alternativas
    ranked_alternatives = global_scores.sort_values(ascending=False)
    #print("\nClasificación de alternativas:")
    #print(ranked_alternatives)

    ### -- Crear DataFrame para clasificación final
    RankFint0 = pd.DataFrame(ranked_alternatives, columns=['Puntuación Global'])
    RankFint0['Alternativa'] = ranked_alternatives.index
    RankFint0.reset_index(drop=True, inplace=True)

    print("\nClasificación Final:")
    print(RankFint0)
    ResultadosMoora = pd.concat([ResultadosMoora,RankFint0], ignore_index=True)

    ### -- PAra valores comparativos en EXCEL
    ValorFin = []
    ValorFin.append(int(RankFint0.iat[0,1]))


    ### -- Asignamos las primeras posiciones de MOOORA como r1 y r2
    Valor1 = int(RankFin.iat[0,1]) # Esta es la primera Mejor alternativa
    Valor2 = int(RankFin.iat[1,1]) # esta es la segunda mejor alternativa
    #print("Valores", Valor1, Valor2)

    # Actualizar los valores para r1 y r2
    r1 = pd.DataFrame(columns=attributes)
    r2 = pd.DataFrame(columns=attributes)
    r1=x.iloc[Valor1]
    r2=x.iloc[Valor2]


    ### -- LOCAL BEST POSITION OF EACH PARTICLE UP TO FIRST ITERATION IS JUST ITS CURRENT POSITION
    # SINCE THER IS NO PREVIUO ITERATION EXISTS
    PBEST=pd.DataFrame(A1t) #Es la primera mejor posición
    print()
    print("pbest(1)=")
    print(PBEST,"\n")


    ### -- GLOBAL BEST FITNESS OF ITERATION #1
    GBF=[]
    pbestt=float(RankFin.iat[0,0])
    GBF.append(pbestt)
    print("GBF(1)=", GBF,"\n")


    ### -- GLOBAL BEST POSITION OF ITERATION 1
    Fx_index=float(RankFin.iat[0,1])
    #print("Fx_index",Fx_index)
    columna = x.iloc[[Fx_index]]
    #print("columna", columna)
    GBP = pd.DataFrame(columna)
    print("gbest(1)=")
    print(GBP)


    Resultados.append(Fx_index+1)
    print()
    print("                 Mejor alternativa= A", Fx_index+1," para la iteración 1")
    print("       ------------------------------------------------------------------------------")
    #print("Resultados de MOORA",ResultadosMoora)



    ### -- Dato para comparativos de EXCEL
    ValorFin.append(Fx_index+1)
    #print("ValorFin",ValorFin)
    #print()
    CMp1 = pd.DataFrame({'MOORA':[ValorFin[0]],'MOORAFIN':[ValorFin[1]]})
    Comparativo = pd.concat([Comparativo,CMp1], ignore_index=True)
    print("Comparativo")
    print(Comparativo)


    # ********************************************************************** ITERACIÓN 2 a N
    t=1
    par=0
    longV1=0
    longseg=5
    iii=0

    while (t<T):

        #print("ENTRE \n")
        print("\n ITERACIÓN #",t+1,"-----------------------","\n")
        print("w(inertia) = ",wwi)
        print("c1 = ",c1)
        print("c2 = ",c2)
        print("No. de iteraciones = ",T,"\n")
        print("r1_new = ",r1,"\n")
        print("r2_new = ",r2,"\n")
        print("Rango de valores: (",rangoMin,",",rangoMax,") \n")
        Fxce=[]
        ii=0
        longVel=a*t
        #print("tr12",tr12)
        r1lt = r1.values.tolist() #Convertir DataFrame a lista de listas
        r2lt = r2.values.tolist() #Convertir DataFrame a lista de listas

        for j in range(a):
            otroV=[]
            otroCP=[]
            CAA=(len(CP)-a)

            for i in range(n): # n=criterios

                # 1-a) ACTUALIZANDO LA VELOCIDAD            
                Vtt1=float(V.iat[CAA,i])
                Vt11=float((wwi*Vtt1))
                #print("    Vt11",round((Vt11),3))

                PBESTtt=float((PBEST.iat[CAA,i]))
                rr1 = r1lt[i]
                CPtt=float((CP.iat[CAA,i])) 
                Vt12=float((c1*rr1*(PBESTtt-CPtt)))
                #print("    Vt12",round((Vt12),3))
                
                GBPtt=float(GBP.iloc[0,i])
                rr2 = r2lt[i]
                Vt13=float((c2*rr2*(GBPtt-CPtt)))          
                #print("    Vt13",round((Vt13),3))
                #print("--- \n")

                VFn=round((float(Vt11+Vt12+Vt13)),3)
                #print("    VFn",round((VFn),3))
                otroV.append(float(VFn))

                # 2-a) ACTUALIZANDO LA PRIMERA  POSICIÓN
                CPtt2=float((CP.iat[CAA,i])) 
                CPFn=round((float(VFn)+float(CPtt2)),3)
                #print("CPFn",round((CPFn),3),"\n ")
                
                # 2-b) Verificar el rango de los valores
                if CPFn<rangoMin: #<-5
                    CPFn=(rangoMin)+.2
                if CPFn>rangoMax: #>5
                    CPFn=(rangoMax)-0.2
                #print(" --- Actualizado CPFn",CPFn)
                #print("+++++++++++++++++++++++++++ \n")
                otroCP.append(float(CPFn))

            V.loc[len(V.index)]=otroV
            CP.loc[len(CP.index)]=otroCP
            CAA=CAA+1
            
        #print("Nueva V = \n", V)
        #print()
        #print("Nueva CP= \n", CP)
            
        ######################################################
        # Calcular el valor de la función objetivo: CURRENT FITNESS (CF = Fx)
        #
        #

        ### --Tomar los últimos 'n' valores del DataFrame x
        xlen2 = CP.iloc[-a:]
        # Reiniciar los índices del DataFrame para que comiencen desde 1
        xlen2 = xlen2.reset_index(drop=True)
        # Asignar los índices de los nuevos renglones
        xlen2.index = candidates[:len(xlen2)]


        ### -- Normalizamos con distancia euclidiana
        # Calcular la suma de los cuadrados de cada fila
        squared_sum2 = (xlen2 ** 2).sum(axis=1)
        #print("squared_sum2",squared_sum2)

        # Calcular la raíz cuadrada de la suma de cuadrados
        norm_factor2 = np.sqrt(squared_sum2)
        #print("norm_factor2",norm_factor2)

        # Normalizar dividiendo cada valor por la raíz cuadrada correspondiente
        normalized_data2 = xlen2.div(norm_factor2, axis=0)
        #print("\n Matriz de datos normalizados (distancia euclidiana):")
        #print(normalized_data2)


        ### -- Ponderamos la matriz normalizada por los pesos de los criterios
        weighted_data2 = normalized_data2 * w
        #print("\nMatriz de datos ponderados:")
        #print(weighted_data2)


        ### -- Cálculo de la puntuación de cada alternativa:
        global_scores2 = [] ## Crear una lista para almacenar los resultados
        for idx, row in weighted_data2.iterrows():
            score2 = 0  # Inicializamos la puntuación para esta fila
            # Iteramos sobre cada valor en la fila y su correspondiente en 'EV'
            for value2, ev_value2 in zip(row, EV):
                # Sumamos o restamos según el valor en 'EV'
                if ev_value2 == "Max":
                    score2 += value2
                else:
                    score2 -= value2
            # Agregamos la puntuación final para esta fila a la lista de resultados
            global_scores2.append(score2)

            #Este valor también corresponde a la PRIMERA mejor posicion local (CF) y Funcion objetivo(Fx)
            CFxe02 = pd.Series(score2)
            CF = pd.concat([CF,CFxe02], ignore_index=True)

            Fx1202 = pd.Series(score2)
            Fx = pd.concat([Fx,Fx1202], ignore_index=True)

        global_scores2 = pd.Series(global_scores2) #Convertir la lista de resultados en una Serie de pandas
        #print("CF NUEVO=")
        #print(CF,"\n")
        #print("LBF=")
        #print(Fx,"\n")
        #print("\nEvaluación de cada alternativa:")
        #print(global_scores2)


        ### -- Clasificación de alternativas
        ranked_alternatives2 = global_scores2.sort_values(ascending=False)
        #print("\nClasificación de alternativas:")
        #print(ranked_alternatives2)


        ### -- Crear DataFrame para clasificación final
        RankFint2 = pd.DataFrame(ranked_alternatives2, columns=['Puntuación Global'])
        RankFint2['Alternativa'] = ranked_alternatives2.index
        RankFint2.reset_index(drop=True, inplace=True)

        print("\nClasificación Final:")
        print(RankFint2)
        ResultadosMoora = pd.concat([ResultadosMoora,RankFint2], ignore_index=True)  
        print()

        
        ### -- PAra valores comparativos en EXCEL
        ValorFin = []
        ValorFin.append(int(RankFint2.iat[0,1])) 

        ### -- Verificamos si la posición actual es mejor que la anterior
        zz1=0
        z1=0
        CFactual=len(CF)-a
        CFAnterior=(CFactual)-a
        par=0
        for j in range(a):
            LxCP=[]
            actual=float(CF.iat[CFactual])
            anterior=float(CF.iat[CFAnterior])

            #print(actual,"<" ,anterior)
            if (actual<anterior) or (actual==anterior): #CP(2)
                for z in range(n):
                    x1=CP.iat[CFactual,z]
                    LxCP.append(round((x1),3))

            else:  # CP(1)
                for z in range(n):
                    x1=CP.iat[CFAnterior,z]
                    LxCP.append(round((x1),3))

            PBEST.loc[len(PBEST.index)]=LxCP
            CFactual=CFactual+1
            CFAnterior=CFAnterior+1

    
        ### -- GLOBAL BEST FITNESS OF ITERATION
        pbestt2=float(RankFint2.iat[0,0])
        GBF.append(pbestt2)
        print("GBF = ", GBF,"\n")

    
        ### -- GLOBAL BEST POSITION OF ITERATION 
        Fx_index2=float(RankFint2.iat[0,1])
        #print("Fx_index2",Fx_index2)
        columna = x.iloc[[Fx_index2]]
        #print("columna", columna)
        GBP = pd.DataFrame(columna)
        print("gbest(1)=")
        print(GBP)


        #Fx_index2=Fx_index2-(a*t)
        print(Fx_index2)
        Resultados.append(Fx_index2+1)
        print("           Mejor alternativa= A", Fx_index2+1," para la iteración 1")
        print("      ---------------------------------------------------------------")
        print()



        ### -- IMPRESIÓN DE RESULTADOS  
        seg=a*t #5*1=5  
        #Mejor=(Fx_index2+1)-(a*t) si fuera CP
        Mejor=(Fx_index+1)
        #print("           Mejor alternativa= A", Mejor," para la iteración",t+1)
        #print("      ---------------------------------------------------------------")
        
        ii=ii+1
        iii=iii+1
        t=t+1



        ### -- Dato para comparativos de EXCEL
        ValorFin.append(Mejor)
        #print("ValorFin",ValorFin)
        #print()
        CMp1 = pd.DataFrame({'MOORA':[ValorFin[0]],'MOORAFIN':[ValorFin[1]]})
        Comparativo = pd.concat([Comparativo,CMp1], ignore_index=True)
        #print("Comparativo")
        #print(Comparativo)    




    print()
    print()
    print("**************************")
    print("Resultados Finales")
    print("**************************")

    RMOORA=[]
    for j in range(a):
        MOORA1=int(RankFin.iat[j,1])
        #print("MM", MOORA1)
        RMOORA.append(MOORA1)
    print("   Resultados preliminares de MOORA ",RMOORA, "\n")
    #print("   Resultados preliminares de MOORA ",ResultadosMoora, "\n")


    print("   Iteración","  Mejor_alternativa")
    print("  ---------------------------------")
    dd=0
    for i in range(T):
        print("       ",i+1,"        ","A",int(Resultados[i]))
    print("  --------------------------------- \n")
    #print("Comparativo \n", Comparativo)



    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)

    # Imprimimos los resultados de tiempo
    print()
    print("Algoritmo MOORA-PSO")
    print("Cantidad de Iteraciones:", t)
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:",ejecut)
    print("  ---------------------------------")
    print()


    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/MOORAPSO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dI={"w(inertia)": [wwi], "c1": [c1], "c2": [c2], "No. de iteraciones":[T]}
    dT= {"Algoritmo": ["MOORA-BA"],
        "Cantidad de Iteraciones": [T],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataI = pd.DataFrame(dI)
    dataGBF = pd.DataFrame(GBF)
    dataGBP = pd.DataFrame(GBP)
    dataResult = pd.DataFrame(Resultados)
    dataResultM = pd.DataFrame(ResultadosMoora)
    dataAlt = pd.DataFrame(RankFin)
    dataw = pd.DataFrame(w)
    dataECA = pd.DataFrame(EV)
    dataND = pd.DataFrame(normalized_data)
    datawd= pd.DataFrame(weighted_data)
    datags= pd.DataFrame(global_scores)
    dataOrig=pd.DataFrame(A1t)
    alternativas = Resultados[-10:]
    hora_fin = datetime.datetime.now()


    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos', index=False)
        dataResultM.to_excel(writer, sheet_name='ResultadosMoora')
        dataResult.to_excel(writer, sheet_name='Resultados')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataI.to_excel(writer, sheet_name='Variables_Iniciales')
        dataw.to_excel(writer, sheet_name='w')
        r1.to_excel(writer, sheet_name='r1')
        r2.to_excel(writer, sheet_name='r2')
        x.to_excel(writer, sheet_name='Posiciones')
        V.to_excel(writer, sheet_name='Velocidades')
        CP.to_excel(writer, sheet_name='CP')
        PBEST.to_excel(writer, sheet_name='PBEST')
        Fx.to_excel(writer, sheet_name='Fx')
        dataGBF.to_excel(writer, sheet_name='GBF')
        dataGBP.to_excel(writer, sheet_name='gbest')
        dataND.to_excel(writer, sheet_name='Matriz_normaliza')
        datawd.to_excel(writer, sheet_name='Matriz_ponderada')
        dataAlt.to_excel(writer, sheet_name='Última_clasificación')

        # Ajustar automáticamente el ancho de las columnas en la hoja 'Tiempos'
        worksheet = writer.sheets['Tiempos']
        for i, col in enumerate(dataT.columns):
            column_len = max(dataT[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_len)
    print(f'Datos guardados en el archivo: {excel_filename}')


    ### -- Guardar los mismos datos en un archivo CSV con el mismo número
    csv_filename = f'{base_filename}_{counter}.csv'
    dataT.to_csv(csv_filename, index=False)
    dataResultM.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataI.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    r1.to_csv(csv_filename, mode='a', index=False)
    r2.to_csv(csv_filename, mode='a', index=False)
    x.to_csv(csv_filename, mode='a', index=False)
    V.to_csv(csv_filename, mode='a', index=False)
    CP.to_csv(csv_filename, mode='a', index=False)
    PBEST.to_csv(csv_filename, mode='a', index=False)
    Fx.to_csv(csv_filename, mode='a', index=False)
    dataGBF.to_csv(csv_filename, mode='a', index=False)
    dataGBP.to_csv(csv_filename, mode='a', index=False)
    dataND.to_csv(csv_filename, mode='a', index=False)
    datawd.to_csv(csv_filename, mode='a', index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)

    print(f'Datos guardados en el archivo CSV: {csv_filename}')
    print()
    
    print("  --------------------------------- \n")
    print('Datos guardados el archivo:MOORA - PSO.xlsx')
    print()
    #Imprimimos los resultados de tiempo
    print("Algoritmo MOORA - PSO")
    print("Cantidad de Iteraciones:", t)
    print ("Datos w",w)
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:",fecha_inicio)
    print("Hora de finalizacion:", hora_fin.time())
    print("Tiempo de ejecucion:", hora_fin-hora_inicio)
    print("")
    print()
    await asyncio.sleep(0.1)
    datosMoorapso = {
            "mejor_alternativa": alternativas,
            "iteraciones": t,
            "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
            "fecha_inicio": fecha_inicio.isoformat(),
            "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
            "tiempo_ejecucion": str(hora_fin - hora_inicio)
        }
    
    return datosMoorapso
