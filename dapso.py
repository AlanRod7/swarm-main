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


async def ejecutar_dapso(w, wwi, c1, c2, T, r1, r2):
    
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]
    ResultadosDA=pd.DataFrame()
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

    #Pesos por cada criterio
    w=[float(0.123),float(0.099),float(0.043),float(0.343),float(0.392)]
    #w=[float(0.2),float(0.2),float(0.2),float(0.2),float(0.2)]
    #w=[float(0.400),float(0.200),float(0.030),float(0.070),float(0.300)]
    #print("Pesos por criterio",w)



    #####################
    # **********************************************************************PRIMERA ITERACIÓN (inicial)
    print("ITERACIÓN # 1 -----------------------")

    ####################
    ## DA

    ### -- Solución ideal
    St=[]
    for j in range(n):
        P1=0
        for i in range(a):
            P1 += round(x.iat[i,j],3)
        P2=round((float(P1)/float(a)),3)
        St.append(round((float(P2)),3))
    S= pd.DataFrame(St, columns=["    Solución ideal"])

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

    ### -- Clasificación de alternativas
    PST = pd.DataFrame(PST, columns=["Índice de similitud"])
    PST.sort_values(by="Índice de similitud", ascending=True, inplace=True) # Ordenar el DataFrame PST por el índice de similitud del menor al mayor

    ### -- Crear DataFrame para clasificación final
    RankFin = pd.DataFrame({
        'Índice de similitud': PST['Índice de similitud'].values,
        'Ranking': PST.index})
    print("\nClasificación Final:")
    print(RankFin)


    ####################
    ## PSO

    ### -- Asignamos las primeras posiciones de DA como r1 y r2
    Valor1 = int(RankFin.iat[0,1]) # Esta es la primera Mejor alternativa
    Valor2 = int(RankFin.iat[1,1]) # esta es la segunda mejor alternativa
    r1 = pd.DataFrame(columns=attributes)
    r2 = pd.DataFrame(columns=attributes)
    r1=x.iloc[Valor1]
    r2=x.iloc[Valor2]


    ### -- Dato para comparativos
    ValorFin = []
    ValorFin.append(int(RankFin.iat[0,1]))
    ValorFin.append(0)
    CMp1 = pd.DataFrame({'DA':[ValorFin[0]],'DAFIN':[ValorFin[1]]})
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

    ### -- Solución ideal
    St=[]
    for j in range(n):
        P1=0
        for i in range(a):
            P1 += round(CP.iat[i,j],3)
        P2=round((float(P1)/float(a)),3)
        St.append(round((float(P2)),3))
    S= pd.DataFrame(St, columns=["    Solución ideal"])

    ### -- Índice de similitud
    CFt=[]
    SI1=[]
    PST=[]
    CF = pd.Series(dtype=float)
    Fx= pd.Series(dtype=float)
    ISSFO = pd.DataFrame(columns=attributes)
    for j in range(a):
        SI1=[]
        ISSFOm1=[]
        for i in range(n):
            dat1= float(CP.iat[j,i])
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
    PST_series = pd.Series(PST)
    CF = pd.concat([CF, PST_series], ignore_index=True)
    Fx = pd.concat([Fx, PST_series], ignore_index=True)

    ### -- Clasificación de alternativas
    PST = pd.DataFrame(PST, columns=["Índice de similitud"])
    PST.sort_values(by="Índice de similitud", ascending=True, inplace=True) # Ordenar el DataFrame PST por el índice de similitud del menor al mayor

    ### -- Crear DataFrame para clasificación final
    RankFint0 = pd.DataFrame({
        'Índice de similitud': PST['Índice de similitud'].values,
        'Ranking': PST.index})
    print("\nClasificación Final:")
    print(RankFint0)

    ResultadosDA = pd.concat([ResultadosDA,(RankFint0+1)], ignore_index=True)

    ### -- PAra valores comparativos en EXCEL
    ValorFin = []
    ValorFin.append(int(RankFint0.iat[0,1]))
    Valor1 = int(RankFin.iat[0,1]) # Esta es la primera Mejor alternativa
    Valor2 = int(RankFin.iat[1,1]) # esta es la segunda mejor alternativa
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
    columna = x.iloc[[Fx_index]]
    GBP = pd.DataFrame(columna)
    print("gbest(1)=")
    print(GBP)


    Resultados.append(Fx_index+1)
    print()
    print("                 Mejor alternativa= A", Fx_index+1," para la iteración 1")
    print("       ------------------------------------------------------------------------------")
    #print("Resultados de DA",ResultadosDA)



    ### -- Dato para comparativos de EXCEL
    ValorFin.append(Fx_index)
    #print("ValorFin",ValorFin)
    #print()
    CMp1 = pd.DataFrame({'DA':[ValorFin[0]],'DAFIN':[ValorFin[1]]})
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

        print("\n ITERACIÓN #",t+1,"-----------------------","\n")
        #print("w(inertia) = ",wwi)
        #print("c1 = ",c1)
        #print("c2 = ",c2)
        #print("No. de iteraciones = ",T,"\n")
        print("r1_new = ",r1,"\n")
        print("r2_new = ",r2,"\n")
        #print("Rango de valores: (",rangoMin,",",rangoMax,") \n")
        Fxce=[]
        ii=0
        longVel=a*t
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

                PBESTtt=float((PBEST.iat[CAA,i]))
                rr1 = r1lt[i]
                CPtt=float((CP.iat[CAA,i])) 
                Vt12=float((c1*rr1*(PBESTtt-CPtt)))
                
                GBPtt=float(GBP.iloc[0,i])
                rr2 = r2lt[i]
                Vt13=float((c2*rr2*(GBPtt-CPtt)))          

                VFn=round((float(Vt11+Vt12+Vt13)),3)
                otroV.append(float(VFn))

                # 2-a) ACTUALIZANDO LA PRIMERA POSICIÓN
                CPtt2=float((CP.iat[CAA,i])) 
                CPFn=round((float(VFn)+float(CPtt2)),3)
                
                # 2-b) Verificar el rango de los valores
                if CPFn<rangoMin: #<-5
                    CPFn=(rangoMin)+.2
                if CPFn>rangoMax: #>5
                    CPFn=(rangoMax)-0.2
                otroCP.append(float(CPFn))

            V.loc[len(V.index)]=otroV
            CP.loc[len(CP.index)]=otroCP
            CAA=CAA+1
            
        #print("Nueva V = \n", V)
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

        ### -- Solución ideal
        St=[]
        for j in range(n):
            P1=0
            for i in range(a):
                P1 += round(xlen2.iat[i,j],3)
            P2=round((float(P1)/float(a)),3)
            St.append(round((float(P2)),3))
        S= pd.DataFrame(St, columns=["    Solución ideal"])

        ### -- Índice de similitud
        CFt=[]
        SI1=[]
        PST=[]
        ISSFO = pd.DataFrame(columns=attributes)
        for j in range(a):
            SI1=[]
            ISSFOm1=[]
            for i in range(n):
                dat1= float(xlen2.iat[j,i])
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
        PST_series = pd.Series(PST)
        CF = pd.concat([CF, PST_series], ignore_index=True)
        Fx = pd.concat([Fx, PST_series], ignore_index=True)

        ### -- Clasificación de alternativas
        PST = pd.DataFrame(PST, columns=["Índice de similitud"])
        PST.sort_values(by="Índice de similitud", ascending=True, inplace=True) # Ordenar el DataFrame PST por el índice de similitud del menor al mayor

        ### -- Crear DataFrame para clasificación final
        RankFint2 = pd.DataFrame({
            'Índice de similitud': PST['Índice de similitud'].values,
            'Ranking': PST.index})
        print("\nClasificación Final:")
        print(RankFint2)
        ResultadosDA = pd.concat([ResultadosDA,(RankFint2+1)], ignore_index=True)
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
        #print("GBF = ", GBF,"\n")

    
        ### -- GLOBAL BEST POSITION OF ITERATION 
        Fx_index2=float(RankFint2.iat[0,1])
        #print("Fx_index2",Fx_index2)
        columna = x.iloc[[Fx_index2]]
        #print("columna", columna)
        GBP = pd.DataFrame(columna)
        print("gbest(1)=")
        print(GBP)
        print()


        Resultados.append(Fx_index2+1)
        print("           Mejor alternativa= A", Fx_index2+1," para la iteración ",t+1)
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
        CMp1 = pd.DataFrame({'DA':[ValorFin[0]],'DAFIN':[ValorFin[1]]})
        Comparativo = pd.concat([Comparativo,CMp1], ignore_index=True)
        #print("Comparativo")
        #print(Comparativo)    




    print()
    print()
    print("**************************")
    print("Resultados Finales")
    print("**************************")

    RDA=[]
    for j in range(a):
        DA1=int(RankFin.iat[j,1])+1
        #print("MM", DA1)
        RDA.append(DA1)
    print("   Resultados preliminares de DA ",RDA, "\n")
    #print("   Resultados preliminares de DA ",ResultadosDA, "\n")


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
    print("Algoritmo DA-PSO")
    print("Cantidad de Iteraciones:", t)
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:", ejecut)
    print()


    ####################################################################################
    ### Para guardar información en archivo de EXCEL

    base_filename = 'Experimentos/DAPSO'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dI={"w(inertia)":[wwi], "c1":[c1], "c2":[c2], "No. de iteraciones":[T]}
    dT = {"Método": ["DA-PSO"],
        "Cantidad de Iteraciones": [T],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }
    dataT = pd.DataFrame(dT)
    dataI = pd.DataFrame(dI)
    dataAlt = pd.DataFrame(RankFin)
    dataw = pd.DataFrame(w)
    dataSI = pd.DataFrame(S)
    dataDIS= pd.DataFrame(PSS)
    dataPS= pd.DataFrame(PST)
    dataResult = pd.DataFrame(Resultados)
    dataResultM = pd.DataFrame(ResultadosDA)
    dataGBF = pd.DataFrame(GBF)
    dataGBP = pd.DataFrame(GBP)
    dataOrig=pd.DataFrame(A1t)

    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos')
        dataResultM.to_excel(writer, sheet_name='ResultadosDA')
        dataResult.to_excel(writer, sheet_name='Resultados')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataI.to_excel(writer, sheet_name='Variables_Iniciales')
        dataw.to_excel(writer, sheet_name='w')
        r1.to_excel(writer, sheet_name='r1')
        r2.to_excel(writer, sheet_name='r2')
        x.to_excel(writer, sheet_name='Psiciones')
        V.to_excel(writer, sheet_name='Velocidades')
        CP.to_excel(writer, sheet_name='CP')
        PBEST.to_excel(writer, sheet_name='PBEST')
        Fx.to_excel(writer, sheet_name='Función_objetivo')
        dataGBF.to_excel(writer, sheet_name='GBF')
        dataGBP.to_excel(writer, sheet_name='gbest')
        dataSI.to_excel(writer, sheet_name='Solución_ideal')
        dataDIS.to_excel(writer, sheet_name='Índice_similitud')
        dataPS.to_excel(writer, sheet_name='Produto_sucesivo')
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
    dataSI.to_csv(csv_filename, mode='a', index=False)
    dataDIS.to_csv(csv_filename, mode='a', index=False)
    dataPS.to_csv(csv_filename, mode='a', index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)
    print(f'Datos guardados en el archivo CSV: {csv_filename}')
    print()
    
    await asyncio.sleep(0.1)
    
    datosDapso = {
        'mejor_alternativa' : alternativas,
        'iteraciones' : T,
        'hora_inicio' : hora_inicio.time().strftime('%H:%M:%S'),
        'fecha_inicio' : fecha_inicio.isoformat(),
        'hora_finalizacion' : hora_fin.time().strftime('%H:%M:%S'),
        'tiempo_ejecucion' : str(hora_fin - hora_inicio)
    }

    return datosDapso


