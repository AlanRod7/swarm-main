# Experimento TOPSIS
# https://www.kaggle.com/code/hungrybluedev/topsis-implementation
#
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# ACtualizado 06/Feb/2023

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
import math
import random
from re import X
import datetime
import asyncio


############################################################
### Pre-requisites
# Los datos dados codificados en vectores y matrices
async def ejecutar_topsispso(w,wwi,c1,c2,T,r1,r2):
    
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]
    ResultadosTOPSIS=pd.DataFrame()
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


    #################################################################
    ### --- Controles iniciales \n" 
    wwi=0.7 # Tener un rango menor ayuda a
    c1=2.5    # Este influye en la pobabilidad hacia
    c2=2.5    # Este influye en la pobabilidad hacia
    rangoMin=0 #este rango de valores
    rangoMax=1  
    T=T    #número de iteraciones para PSO
    dim=n*a #dimensión del enjambre

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


    #####################
    # **********************************************************************PRIMERA ITERACIÓN (inicial)
    print("ITERACIÓN # 1 -----------------------")

    ####################
    ## TOPSIS

    ### -- PAso 1 - Normalizando las calificaciones
    normalized_matrix = x.copy()
    for i in range(len(attributes)):
        if i in benefit_attributes:
            normalized_matrix.iloc[:, i] = x.iloc[:, i] / np.linalg.norm(x.iloc[:, i])
        else:
            normalized_matrix.iloc[:, i] = x.iloc[:, i] / np.linalg.norm(x.iloc[:, i], ord=1)
    #print(normalized_matrix)

    ### -- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
    #            Multiplicación de la matriz normalizada por los pesos
    weighted_matrix = normalized_matrix * weights
    #print(weighted_matrix)

    ### -- Paso 3: Identificar las soluciones ideal y anti-ideal
    ideal_best = weighted_matrix.max()
    ideal_worst = weighted_matrix.min()
    #print("Ideal:")
    #print(ideal_best)
    #print("\nAnti-Ideal:")
    #print(ideal_worst)

    ### -- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
    # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
    s_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    #print("Distancias a la solución ideal:")
    #print(s_best)
    #print("\nDistancias a la solución anti-ideal:")
    #print(s_worst)

    ### -- Pasos 5: Cálculo de la puntuación de proximidad relativa
    performance_score = s_worst / (s_best + s_worst)
    #print("Puntuación de proximidad relativa:")
    #print(performance_score)

    ### -- Paso 6: Clasificación de las alternativas
    ranked_candidates = performance_score.sort_values(ascending=True)
    #print("Ranking de los candidatos:")
    #print(ranked_candidates)



    ### -- Crear DataFrame para clasificación final
    RankFin = pd.DataFrame(ranked_candidates, columns=['Puntuación Global'])
    RankFin['Alternativa'] = ranked_candidates.index
    RankFin.reset_index(drop=True, inplace=True)
    print("\nClasificación Final:")
    print(RankFin)


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
    #print("r1",r1)
    #print("r2",r2)


    ### -- Dato para comparativos
    ValorFin = []
    ValorFin.append(int(RankFin.iat[0,1]))
    ValorFin.append(0)
    CMp1 = pd.DataFrame({'TOPSIS':[ValorFin[0]],'TOPSISFIN':[ValorFin[1]]})
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

    #---PAso 1 - Normalizando las calificaciones
    normalized_matrix = CP.copy()
    for i in range(len(attributes)):
        if i in benefit_attributes:
            normalized_matrix.iloc[:, i] = CP.iloc[:, i] / np.linalg.norm(CP.iloc[:, i])
        else:
            normalized_matrix.iloc[:, i] = CP.iloc[:, i] / np.linalg.norm(CP.iloc[:, i], ord=1)
    #print(normalized_matrix)


    #--- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
    #            Multiplicación de la matriz normalizada por los pesos
    weighted_matrix = normalized_matrix * weights
    #print(weighted_matrix)


    #--- Paso 3: Identificar las soluciones ideal y anti-ideal
    ideal_best = weighted_matrix.max()
    ideal_worst = weighted_matrix.min()
    #print("Ideal:")
    #print(ideal_best)
    #print("\nAnti-Ideal:")
    #print(ideal_worst)


    #--- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
    # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
    s_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    #print("Distancias a la solución ideal:")
    #print(s_best)
    #print("\nDistancias a la solución anti-ideal:")
    #print(s_worst)


    #--- Pasos 5: Cálculo de la puntuación de proximidad relativa
    CF = pd.Series(dtype=float)
    Fx= pd.Series(dtype=float)
    performance_score = s_worst / (s_best + s_worst)
    #print("Puntuación de proximidad relativa:")
    #print(performance_score)
    CF = pd.concat([CF,performance_score], ignore_index=True)
    Fx = pd.concat([Fx,performance_score], ignore_index=True)
    #print("CF NUEVO=")
    #print(CF,"\n")
    #print("LBF=")
    #print(Fx,"\n")


    #--- Paso 6: Clasificación de las alternativas
    ranked_candidates = performance_score.sort_values(ascending=True)
    #print("Ranking de los candidatos:")
    #print(ranked_candidates)

    ### -- Crear DataFrame para clasificación final
    RankFint0 = pd.DataFrame(ranked_candidates, columns=['Puntuación Global'])
    RankFint0['Alternativa'] = ranked_candidates.index
    RankFint0.reset_index(drop=True, inplace=True)
    print("\nClasificación Final:")
    print(RankFint0)
    ResultadosTOPSIS = pd.concat([ResultadosTOPSIS,RankFint0], ignore_index=True)

    ### -- PAra valores comparativos en EXCEL
    ValorFin = []
    ValorFin.append(int(RankFint0.iat[0,1]))


    ### -- Asignamos las primeras posiciones de MOOORA como r1 y r2
    Valor1 = int(RankFint0.iat[0,1]) # Esta es la primera Mejor alternativa
    Valor2 = int(RankFint0.iat[1,1]) # esta es la segunda mejor alternativa
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
    #print("Resultados de TOPSIS",ResultadosTOPSIS)



    ### -- Dato para comparativos de EXCEL
    ValorFin.append(Fx_index+1)
    #print("ValorFin",ValorFin)
    #print()
    CMp1 = pd.DataFrame({'TOPSIS':[ValorFin[0]],'TOPSISFIN':[ValorFin[1]]})
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

        print("\n")
        print("########################################################")
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
        #print("xlen2")
        #print(xlen2)

        #---PAso 1 - Normalizando las calificaciones
        normalized_matrix2 = xlen2.copy()
        for i in range(len(attributes)):
            if i in benefit_attributes:
                normalized_matrix2.iloc[:, i] = xlen2.iloc[:, i] / np.linalg.norm(xlen2.iloc[:, i])
            else:
                normalized_matrix2.iloc[:, i] = xlen2.iloc[:, i] / np.linalg.norm(xlen2.iloc[:, i], ord=1)
        #print(normalized_matrix2)


        #--- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
        #            Multiplicación de la matriz normalizada por los pesos
        weighted_matrix2 = normalized_matrix2 * weights
        #print(weighted_matrix2)


        #--- Paso 3: Identificar las soluciones ideal y anti-ideal
        ideal_best2 = weighted_matrix2.max()
        ideal_worst2 = weighted_matrix2.min()
        #print("Ideal:")
        #print(ideal_best2)
        #print("\nAnti-Ideal:")
        #print(ideal_worst2)


        #--- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
        # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
        s_best2 = np.sqrt(((weighted_matrix2 - ideal_best2) ** 2).sum(axis=1))
        s_worst2 = np.sqrt(((weighted_matrix2 - ideal_worst2) ** 2).sum(axis=1))
        #print("Distancias a la solución ideal:")
        #print(s_best2)
        #print("\nDistancias a la solución anti-ideal:")
        #print(s_worst2)


        #--- Pasos 5: Cálculo de la puntuación de proximidad relativa
        performance_score2 = s_worst2 / (s_best2 + s_worst2)
        #print("Puntuación de proximidad relativa:")
        #print(performance_score2)
        CF = pd.concat([CF,performance_score2], ignore_index=True)
        Fx = pd.concat([Fx,performance_score2], ignore_index=True)
        #print("CF NUEVO=")
        #print(CF,"\n")
        #print("LBF=")
        #print(Fx,"\n")


        #--- Paso 6: Clasificación de las alternativas
        ranked_candidates2 = performance_score2.sort_values(ascending=True)
        #print("Ranking de los candidatos:")
        #print(ranked_candidates2)


        ### -- Crear DataFrame para clasificación final
        RankFint2 = pd.DataFrame(ranked_candidates2, columns=['Puntuación Global'])
        RankFint2['Alternativa'] = ranked_candidates2.index
        RankFint2.reset_index(drop=True, inplace=True)
        print("\nClasificación Final:")
        print(RankFint2)
        ResultadosTOPSIS = pd.concat([ResultadosTOPSIS,RankFint2], ignore_index=True)
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
        CMp1 = pd.DataFrame({'TOPSIS':[ValorFin[0]],'TOPSISFIN':[ValorFin[1]]})
        Comparativo = pd.concat([Comparativo,CMp1], ignore_index=True)
        #print("Comparativo")
        #print(Comparativo)    




    print()
    print()
    print("**************************")
    print("Resultados Finales")
    print("**************************")

    RTOPSIS=[]
    for j in range(a):
        TOPSIS1=int(RankFin.iat[j,1])
        #print("MM", TOPSIS1)
        RTOPSIS.append(TOPSIS1)
    print("   Resultados preliminares de TOPSIS ",RTOPSIS, "\n")
    #print("   Resultados preliminares de TOPSIS ",ResultadosTOPSIS, "\n")


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
    alternativas = Resultados[-10:]

    # Imprimimos los resultados de tiempo
    print("Algoritmo TOPSIS-PSO")
    print("Cantidad de Iteraciones:", t)
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:", ejecut)
    print()



    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/TOPSISPSO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dI={"w(inertia)": [wwi], "c1":[c1], "c2":[c2], "No. de iteraciones":[T]}
    dT= {"Algoritmo": ["TOPSIS-PSO"],
        "Cantidad de Iteraciones": [T],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataI = pd.DataFrame(dI)
    dataMOO=pd.DataFrame(ResultadosTOPSIS)
    dataResult = pd.DataFrame(Resultados)
    dataOrig=pd.DataFrame(A1t)
    dataw = pd.DataFrame(w)
    dataBen=pd.DataFrame(benefit_attributes)
    dataNMx=pd.DataFrame(normalized_matrix)
    dataNMxW=pd.DataFrame(weighted_matrix)
    dataIb=pd.DataFrame(ideal_best)
    dataIw=pd.DataFrame(ideal_worst)
    dataSBt=pd.DataFrame(s_best)
    dataSwT=pd.DataFrame(s_worst)
    datapsc=pd.DataFrame(performance_score)
    dataRkf1 = pd.DataFrame(RankFin)
    dataRkf2 = pd.DataFrame(RankFint0)
    dataGBF = pd.DataFrame(GBF)
    dataGBP = pd.DataFrame(GBP)
    dataComp= pd.DataFrame(Comparativo)

    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos')
        dataMOO.to_excel(writer, sheet_name='ResultadosTOPSIS')
        dataResult.to_excel(writer, sheet_name='Resultados')
        dataComp.to_excel(writer, sheet_name='Compara_TOPSIS')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataI.to_excel(writer, sheet_name='Valores_Iniciales')
        dataw.to_excel(writer, sheet_name='w')
        dataBen.to_excel(writer, sheet_name='Atributos_beneficios')
        r1.to_excel(writer, sheet_name='r1')
        r2.to_excel(writer, sheet_name='r2')
        x.to_excel(writer, sheet_name='Pocisiones')
        V.to_excel(writer, sheet_name='Velocidades')
        CP.to_excel(writer, sheet_name='Posiciones_locales')
        PBEST.to_excel(writer, sheet_name='PBEST')
        Fx.to_excel(writer, sheet_name='Función_objetivo')
        dataGBF.to_excel(writer, sheet_name='GBF')
        dataGBP.to_excel(writer, sheet_name='gbest')
        dataNMx.to_excel(writer, sheet_name='Matriz_normalizada')
        dataNMxW.to_excel(writer, sheet_name='Matriz_normalizada_xPesos')
        dataIb.to_excel(writer, sheet_name='Solución_ideal(SI)')
        dataIw.to_excel(writer, sheet_name='Solución_anti-ideal(SaI)')
        dataSBt.to_excel(writer, sheet_name='Distancia_SI')
        dataSwT.to_excel(writer, sheet_name='Distancia_SaI')
        datapsc.to_excel(writer, sheet_name='Puntuación_prox_relativa')
        dataRkf1.to_excel(writer, sheet_name='Ranking_alternativas')
        dataRkf2.to_excel(writer, sheet_name='Ranking_alternativas_nw')
        

        # Ajustar automáticamente el ancho de las columnas en la hoja 'Tiempos'
        worksheet = writer.sheets['Tiempos']
        for i, col in enumerate(dataT.columns):
            column_len = max(dataT[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_len)
    print(f'Datos guardados en el archivo: {excel_filename}')

    ### -- Guardar los mismos datos en un archivo CSV con el mismo número
    csv_filename = f'{base_filename}_{counter}.csv'
    dataI.to_csv(csv_filename, mode='a', index=False)
    dataT.to_csv(csv_filename, index=False)
    dataT.to_csv(csv_filename, mode='a', index=False)
    dataMOO.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    dataComp.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataI.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    dataBen.to_csv(csv_filename, mode='a', index=False)
    r1.to_csv(csv_filename, mode='a', index=False)
    r2.to_csv(csv_filename, mode='a', index=False)
    x.to_csv(csv_filename, mode='a', index=False)
    V.to_csv(csv_filename, mode='a', index=False)
    CP.to_csv(csv_filename, mode='a', index=False)
    PBEST.to_csv(csv_filename, mode='a', index=False)
    Fx.to_csv(csv_filename, mode='a', index=False)
    dataGBF.to_csv(csv_filename, mode='a', index=False)
    dataGBP.to_csv(csv_filename, mode='a', index=False)
    dataNMx.to_csv(csv_filename, mode='a', index=False)
    dataNMxW.to_csv(csv_filename, mode='a', index=False)
    dataIb.to_csv(csv_filename, mode='a', index=False)
    dataIw.to_csv(csv_filename, mode='a', index=False)
    dataSBt.to_csv(csv_filename, mode='a', index=False)
    dataSwT.to_csv(csv_filename, mode='a', index=False)
    datapsc.to_csv(csv_filename, mode='a', index=False)
    dataRkf1.to_csv(csv_filename, mode='a', index=False)
    dataRkf2.to_csv(csv_filename, mode='a', index=False)
    print(f'Datos guardados en el archivo CSV: {csv_filename}')
    print()
    
    await asyncio.sleep(0.1)
    datosTopsispso = {
        "mejor_alternativa": alternativas,
        "iteraciones": t,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }
    
    return datosTopsispso