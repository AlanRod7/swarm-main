
# Experimento TOPSIS-BA
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# Actualización 24-Feb-2024


from flask import Flask
import random
from decimal import Decimal
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import os
import xlsxwriter
import matplotlib.pyplot as plt
import math
import datetime
from math import e
import asyncio


async def ejecutar_topsisba(w, alphaBa, gamma, iter_max):

    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    Resultados=[]

    ######################################################################################################################################################
    # Paso # 1: Inicialice la primera posición y velocidad
    #
    #
    n=9 #Alternativas
    d=5 # Criterios


    #-------------- Posición inicial
    #               lugar donde algo se encuentra, es decir la posición de los mirciélagos
    #
    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión")
    attributes = ["C1", "C2", "C3", "C4", "C5"]
    candidates = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

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
    print(x,"\n")


    #-------------- Velocidad inicial
    data = np.random.uniform(0, 1, size=(n, d)).round(3)
    v = pd.DataFrame(data, columns=attributes, index=candidates)
    print("\n Velocidad inicial=")
    print(v,"\n")

    #############################################################################################
    print("\n-------------------------------------------")
    print("Controles iniciales")

    # Contien las evaluaciones cardinales de cada alternativa respecto a cada criterio
    EV = ["Min", "Min", "Min", "Min", "Min"]
    #EV = ["Max","Max","Max","Max","Max"]
    #print("Evaluaciones cardinales de cada alternativa respecto a cada criterio:")
    #print(EV,"             \n")

    # Los índices de los atributos (de base cero) que se consideran beneficiosos.
    # Se supone que los índices no mencionados son atributos de costos.
    benefit_attributes = set([0, 1, 2, 3, 4])

    ### -- Pesos por cada criterio
    #w = [0.400, 0.200, 0.030, 0.070, 0.300]
    #w = [0.300, 0.200, 0.200, 0.150, 0.150] 
    #w = [0.200, 0.200, 0.200, 0.200, 0.200]
    #w = [0.123, 0.099, 0.043, 0.343, 0.392]
    weights = pd.Series(w, index=attributes)
    #print(weights,"          \n")

    alpha=alphaBa   # Valor para actualizar Ai Loudness 
    gamma=gamma  # Valor para actualizar ri Pulse Rate

    # ----Iteraciones
    #iter_max = int(input("\n""Ingrese el numero de iteraciones maximas: \n"))
    iter_max = iter_max

    # ----Tasa de pulso (Pulse rate) 
    #Valores para tasa de pulso [0,1]
    #print("Ingrese los valores iniciales para ri / Pulse emission: [0,1]")
    ri=[]  
    ri = pd.Series(np.random.rand(n))
    ri_ini=ri
    #print(ri)

    # ----Sonoridad (Loudness)
    #         Característica de sonido y significa cuán fuerte o suave es el sonido al oyente y denotaremos la sonoridad con Ai
    #         Valores sonoridad [1,2]
    #print("\n""Ingrese los valores iniciales para la sonoridad Ai / Loudness: [1,2]")
    ai=[]   
    ai = pd.Series(np.random.uniform(1,2, size=n))
    ai_ini=ai
    #ai=pd.Series(ai)
    #print(ai)

    # ---- Frecuencia   
    #         Ondas producidas por unidad de tiempo y esta es la longitud de onda que es la distancia mínima entre dos partículas (Wavelength)
    #         Valores para frecuencia [0,2]
    #print("\n""Ingrese los valores iniciales para fi / Frequency: [0,2]")
    f=[] 
    f01=[]
    for c in range (n):
        #f0 = float(input("BAT" + " " + str(c+1) + "\n"))
        f0 = 0
        f01.append(f0)
    f = pd.DataFrame({'C1':[f01[0]],'C2':[f01[1]],'C3':[f01[2]],'C4':[f01[3]],'C5':[f01[4]]})
    #f = pd.Series(np.random.uniform(0,2, size=n))
    #print(f)
    # Parametros (variables adicionales / Segunda iteracion en delante)
    #temporary_position=np.zeros((n,d))  # Variable para actualizar posicion
    fmin=0  # Frecuencia minima 
    fmax=1  # Frecuencia maxima

    # Valores Aleatorios
    #print("Los valores aleatorios usados, son los siguientes:")
    #rnd=pd.Series([0.1592, 0.1844, 0.0880, 0.0707])
    #rnd=pd.Series([0.1592, 0.1844, 0.0880, 0.0707, 0.5962, 0.8244, 0.8580, 0.0087])
    rnd=pd.Series(np.random.rand(n))
    #print(rnd)


    print("             alpha",alpha) 
    print("             gamma",gamma)
    print("             Número de iteraciones:",iter_max )
    print("\n Evaluaciones cardinales de cada alternativa:")
    print("     ",EV)
    print("\n Pesos por cada criterio")
    print("     ", w)
    print("\n Tasa de pulso (Pulse rate)")
    print(ri)
    print("\n Sonoridad (Loudness)")
    print(ai)
    print("\n Frecuencia")
    print(f)
    print("\n Valores Aleatorios")
    print(rnd)
    #print("--------------------------------------------------  \n")


    ######################################################################################################################################################
    ################################# Primera iteracion

    # Iteración
    it = 0
    ResultadosTOPSIS = []
    global_best=pd.DataFrame(columns=attributes)
    CCB = pd.DataFrame(columns=attributes)

    while it < iter_max:
        print('\n =======================================================')
        print("                    ITERACIÓN #",it)
        print('=======================================================')

        # Tomar los últimos 'n' valores del DataFrame x
        xlen = x.iloc[-n:]
        # Reiniciar los índices del DataFrame para que comiencen desde 1
        xlen = xlen.reset_index(drop=True)
        # Asignar los índices de los nuevos renglones
        #xlen.index = candidates[:len(xlen)]
        print(xlen)

        ######################################################
        # Calcular el valor de la función objetivo
        #
        #

        #---PAso 1 - Normalizando las calificaciones
        normalized_matrix = xlen.copy()
        for i in range(len(attributes)):
            if i in benefit_attributes:
                normalized_matrix.iloc[:, i] = xlen.iloc[:, i] / np.linalg.norm(xlen.iloc[:, i])
            else:
                normalized_matrix.iloc[:, i] = xlen.iloc[:, i] / np.linalg.norm(xlen.iloc[:, i], ord=1)
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
        performance_score = s_worst / (s_best + s_worst)
        #print("Puntuación de proximidad relativa:")
        #print(performance_score)

        #--- Paso 6: Clasificación de las alternativas
        ranked_candidates = performance_score.sort_values(ascending=True)
        #print("Ranking de los candidatos:")
        #print(ranked_candidates)


        
        ### -- Crear DataFrame para clasificación final
        RankFin = pd.DataFrame(ranked_candidates, columns=['Puntuación Global'])
        RankFin['Alternativa'] = ranked_candidates.index
        RankFin.reset_index(drop=True, inplace=True)
        print("\nClasificación Final:")
        print(RankFin)

        
        # Mejor posición local
        # 1) 
        FuncObj=RankFin['Puntuación Global'].round(6).tolist() 
        #print("\n --------------------------------------------------")
        #print("  Función objetivo= \n",FuncObj)
        # 2) se asigna el mejor función objetivo
        #IF_maxt=int(RankFin.iloc[0]['Alternativa'])
        IF_maxt=(RankFin.iat[0,1])# guarda la alternativa, para usar el index en la matriz original (este listado inicia en 0, por lo que no se requiere restar un 1)
        Mejor=IF_maxt 
        #print("\n Index de la Función objetivo(min)= \n",IF_maxt) # los valores inician en 0

        print("\n--------------------------------------------------")
        print("La mejor solución es la alternativa: A", (RankFin.iloc[0]['Alternativa'])+1, "con una puntuación global de:", RankFin.round(6).iloc[0]['Puntuación Global'])
        
        #print("xlen",xlen)
        selected_row = xlen.iloc[IF_maxt]
        #print(selected_row.tolist())
        global_best =pd.DataFrame(columns=attributes)
        global_best = selected_row
        #print("\nEl mejor global local: ", global_best.tolist(), "\n")
        print("\nEl mejor global local: ")
        print(global_best, "\n")

        # Agregar los valores de TOPSIS, solo en la iteración 1  
        if it==0:
            ResultadosTOPSIS = RankFin['Alternativa'].tolist()  # Redondear a 6 decimales
            #print("Resultados preliminares de TOPSIS:")
            #print(ResultadosTOPSIS)

        #Result.append(Mejor+1)
        Resultados.append(IF_maxt)
        #print(Resultados)


        ######################################################
        # Actualizar Velocidad:
        #
        #
        # 1) definir la frecuencia de cada murciélago
        #print ("Frecuencia actual", f)
        fxVVw=[]
        for j in range(d):

            # Beta es un valor aleatorio, de una distribución uniforme [0,1]
            beta = round((random.uniform(0,1)),3)
            fxxx = round((fmin + (fmax-fmin)*beta),3)
            fxVVw.append(round((float(fxxx)),3))
            
            #print("     beta= ", beta)
            #print("     fmin= ",fmin)
            #print("     fmax= ",fmax)
            #print("      Nueva_frecuencia=",fxxx)
            #print()

        f1 = pd.DataFrame({'C1':[fxVVw[0]],'C2':[fxVVw[1]],'C3':[fxVVw[2]],'C4':[fxVVw[3]],'C5':[fxVVw[4]]})
        f = pd.concat([f,f1], ignore_index=True)
        print("\n -------------------------------------------------------------")
        print("\n Nuevas frecuencias: ")
        f_actual=len(f)-1
        f_fin=len(f)
        print(f.iloc[f_actual:f_fin])
        #print(f)
        print()

        # 2) Actualizar la velocidad
        # Vi = Velocidad inicial + (Posición actual - Mejor_posición) * Frecuencia
        v_actual=len(v)-n
        x_actual=len(x)-n
        #print("x", x)
        #print("x", x_actual)
        #print("v", v)
        #print("v", v_actual)
        #print("global_best")
        #print(global_best)
        
        for i in range(n):#n alternativas
            VAct=[]
            f_actual=len(f)-1
            gb_actual=len(global_best)-1

            for j in range(d): #dcriterios
                # Vi = Velocidad inicial + (Posición actual - Mejor_posición) * Frecuencia
                
                #print("     x.iat[i,j]",x.iat[x_actual,j])
                #print("     v.iat[i,j]",v.iat[v_actual,j])
                #print("     f.iat[f_actual,i]",f.iat[f_actual,j])
                #print("     global_best.iat[f_actual,i]",global_best.iat[j])

                VAct1 = v.iat[v_actual,j] + (x.iat[x_actual,j] - global_best.iat[j]) * f.iat[f_actual,j]
                VAct.append(round((float(VAct1)),3))
                        
                #print(v.iat[j,v_actual],"+ (", x.iat[j,x_actual], "-", global_best.iat[global_actual,i], ")*", f.iat[f_actual,i])
                #print("   VAct===", VAct1)
                #print("---")

            x_actual=x_actual+1
            v_actual=v_actual+1
            #print("v_new",v_actual)
            Ver1 = pd.DataFrame({'C1':[VAct[0]],'C2':[VAct[1]],'C3':[VAct[2]],'C4':[VAct[3]],'C5':[VAct[4]]})
            v = pd.concat([v,Ver1], ignore_index=True)
        
        print("\n velocidad actualizada")
        #print(v)
        v_actual=len(v)-n
        v_fin=len(v)
        print(v.iloc[v_actual:v_fin])

        ######################################################
        # Actualizar posición
        #
        #
        # Xi = posición_inicial + Velociad_actualizada
        v_actual=len(v)-n
        x_actual=len(x)-n

        for i in range(n):#n alternativas
            XAct=[]
            for j in range(d): #dcriterios
                #print("posición",x.iat[x_actual,j])
                #print("vel_actual",v.iat[v_actual,j])
                XAct1 = x.iat[x_actual,j] + v.iat[v_actual,j] 
                XAct.append(round((float(XAct1)),3))
            #print("XAct", XAct)
            #print("-------")

            x_actual += 1
            v_actual += 1

            Ver2 = pd.DataFrame({'C1':[XAct[0]],'C2':[XAct[1]],'C3':[XAct[2]],'C4':[XAct[3]],'C5':[XAct[4]]})
            # Ver2 = pd.DataFrame({f'C{i+1}': [XAct[i] for i in range(d)]})
            x = pd.concat([x,Ver2], ignore_index=True)
        print("\n Posición actualizada")
        #print(x)
        x_actual=len(x)-n
        x_fin=len(x)
        print(x.iloc[x_actual:x_fin])
        #print(x.iloc[9:18])



        ######################################################
        # Verifique si (Rand > ri)
        #
        #

        # Volumne medio de todos los murciélagos, es decir el promedio de posiciones actuales
        # 1) Seleccionamos los últimos elementos de la serie
        ultimos__elementos = ai[-n:]
            #print("ultimos elementos", ultimos__elementos)
        # 2) Calculamos el promedio de los últimos elementos
        rrtP1 = sum(ultimos__elementos) / len(ultimos__elementos)
            #print("Promedio Fuerza de sonido")
            #print(rrtP1)

        if all(rnd > ri):
        #if(0.1>0.818): #Valor para pruebas de código
            print("\n Generar una solución local a través del paseo aleatorio")
            
            x_actual=len(x)-n
            for i in range(n):
                xx1=[]
                for j in range(d):

                    # Posición
                    rrtP2 = float(x.iat[x_actual,j])
                    #print("      Posición = ", rrtP2)

                    # Aleatorio en un rango de [-1,1]
                    aleatorio2 = round((random.uniform(-1,1)),3)
                    #print("    Aleatorio2 = ", aleatorio2)

                    # Volumen medio de todos los murciélagos, es decir el promedio de posiciones actuales
                    #print("Promedio de la Fuerza de sonido")
                    #print(rrtP1)

                    # Nueva posición
                    rrtP3 =round((rrtP2 + aleatorio2 *rrtP1),3)
                    xx1.append(round((float(rrtP3)),3))
                    #print("                  x[i]= ",rrtP3)             
                
                x_actual=x_actual+1
                #print("x_newFIN",xx1)
                Xer1 = pd.DataFrame({'C1':[xx1[0]],'C2':[xx1[1]],'C3':[xx1[2]],'C4':[xx1[3]],'C5':[xx1[4]]})
                x = pd.concat([x,Xer1], ignore_index=True)
            #print(x)
            print("Nuevas posiciones")
            xal=len(x)-n
            print(x.iloc[xal:xal+n])
        else:
            #print("\n ***************************************************************ENtre al ELSE")
            x_actual=len(x)-n
            v_actual=len(v)-n
            f_actual=len(f)-1

            for i in range(n):
                xx1=[]
                xx2=[]
                for j in range(d):
                    
                    # velocidad
                    rrtP4 = float(v.iat[v_actual,j])
                    #print("     Velocidad = ",rrtP4)

                    # Posición
                    rrtP3 = float(x.iat[x_actual,j])
                    #print("      Posición = ",rrtP3)

                    #global_best
                    rrtP7 = float(global_best.iat[j])
                    #print("   global_best = ",rrtP7)

                    #Nueva_frecuencia
                    rrtP8 = float(f.iat[f_actual,j])
                    #print("    Nueva_frec = ",rrtP8)

                    #nueva velocidad
                    rrtP5 = round((rrtP4 + (rrtP3 -rrtP7) * rrtP8),3)
                    #print("      Nueva_Vel = ",rrtP5)
                    
                    # nueva posición
                    rrtP6 =round((rrtP3 + rrtP5),3)
                    #print(" Nueva_posición = ",rrtP6)

                    xx1.append(round((float(rrtP6)),3))
                    xx2.append(round((float(rrtP5)),3))

                #print("x_newFIN",xx1)
                #print("v_newFIN",xx2)
                x_actual += 1
                v_actual += 1
                Xer1 = pd.DataFrame({'C1':[xx1[0]],'C2':[xx1[1]],'C3':[xx1[2]],'C4':[xx1[3]],'C5':[xx1[4]]})
                x = pd.concat([x,Xer1], ignore_index=True)
                Ver1 = pd.DataFrame({'C1':[xx2[0]],'C2':[xx2[1]],'C3':[xx2[2]],'C4':[xx2[3]],'C5':[xx2[4]]})
                v = pd.concat([v,Ver1], ignore_index=True)
            print("\n Nueva posición generada")
            xal=len(x)-n
            print(x.iloc[xal:xal+n])
            print("\n Nueva velocidad generada")
            val=len(v)-n
            print(v.iloc[val:val+n])

        ######################################################
        # Selección de condiciones para actualizar: La tasa de pulso(ri) y la Sonoridad(A)
            # If (rand > Ai and f(new_fitness < f(Best_fitness))
            # if all(rnd>ri) and all(FuncObj[1]< FuncObj[0])

        IFmin_cont =len(FuncObj)-1
        #print("IFmin_cont",IFmin_cont)

    
        ######################################################
        # Calcular la NUEVA función objetivo: Calcular el valor de la función objetivo
        #
        #
        xNW= x.iloc[xal:xal+n]
        # Reiniciar los índices del nuevo DataFrame para que comiencen desde 1
        xNW = xNW.reset_index(drop=True)
        # Asignar los índices de los nuevos renglones
        #xNW.index = candidates[:len(xNW)]
        #print("xNW",xNW)


        #---PAso 1 - Normalizando las calificaciones
        normalized_matrixNW = xNW.copy()
        for i in range(len(attributes)):
            if i in benefit_attributes:
                normalized_matrixNW.iloc[:, i] = xNW.iloc[:, i] / np.linalg.norm(xNW.iloc[:, i])
            else:
                normalized_matrixNW.iloc[:, i] = xNW.iloc[:, i] / np.linalg.norm(xNW.iloc[:, i], ord=1)
        #print(normalized_matrixNW)

        #--- PAso 2: Cálculo de las calificaciones normalizadas ponderadas
        #            Multiplicación de la matriz normalizada por los pesos
        weighted_matrixNW = normalized_matrixNW * weights
        #print(weighted_matrixNW)

        #--- Paso 3: Identificar las soluciones ideal y anti-ideal
        ideal_bestNW = weighted_matrixNW.max()
        ideal_worstNW = weighted_matrixNW.min()
        #print("Ideal:")
        #print(ideal_bestNW)
        #print("\nAnti-Ideal:")
        #print(ideal_worstNW)

        #--- Pasos 4: Cálculo de las distancias a las soluciones ideal y anti-ideal
        # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
        s_bestNW = np.sqrt(((weighted_matrixNW - ideal_bestNW) ** 2).sum(axis=1))
        s_worstNW = np.sqrt(((weighted_matrixNW - ideal_worstNW) ** 2).sum(axis=1))
        #print("Distancias a la solución ideal:")
        #print(s_bestNW)
        #print("\nDistancias a la solución anti-ideal:")
        #print(s_worstNW)

        #--- Pasos 5: Cálculo de la puntuación de proximidad relativa
        performance_scoreNW = s_worstNW / (s_bestNW + s_worstNW)
        #print("Puntuación de proximidad relativa:")
        #print(performance_scoreNW)

        #--- Paso 6: Clasificación de las alternativas
        ranked_alternativesNW = performance_scoreNW.sort_values(ascending=True)
        #print("\nClasificación de alternativas:")
        #print(ranked_alternativesNW)    


        
        ### -- Crear DataFrame para clasificación final
        RankFinNW = pd.DataFrame(ranked_alternativesNW, columns=['Puntuación Global'])
        RankFinNW['Alternativa'] = ranked_alternativesNW.index
        RankFinNW.reset_index(drop=True, inplace=True)
        print("\nClasificación Final:")
        print(RankFinNW)


        # Mejor posición local
        # 1) 
        FuncObjN=RankFinNW['Puntuación Global'].round(6).tolist() 
        print("\n Nueva Función objetivo= \n",FuncObjN)
        # 2) se asigna el mejor función objetivo
        IF_maxtt=int(RankFinNW.iloc[0]['Alternativa'])
        MejorNW=IF_maxtt
        print("\n Función objetivo(min)= \n",IF_maxtt)
        
        #print("",xNW)
        selected_rowNW = xNW.iloc[[IF_maxtt]]
        #print(selected_rowNW)
        global_bestNW =pd.DataFrame(columns=attributes)
        global_bestNW= selected_rowNW
        print("\nNuevo mejor global local: ")
        print(global_bestNW, "\n")

    

        
        # Mejor posición local y Función objetivo ( anterior y nueva)
        print("\n -----------------------------------------------------------------------------------")
        print("Función objetivo= ")
        print("                        ",FuncObj)
        print("Nueva Función objetivo= ")
        print("                        ",FuncObjN)
        print()
        print("Posición local= ")
        print("                       ",IF_maxt)
        print("Nueva posición local ")
        print("                       ",IF_maxtt)
        print(" -----------------------------------------------------------------------------------")

        


        # Teniendo la nueva función objetivo, ya se puede ver si se actualiza o no la tasa de pulso(ri) y la Sonoridad(A)
        ai_actual=len(ai)-n
            #print(ai.iloc[ai_actual:ai_actual+n])
        ri_actual=len(ri)-n
            #print(ri.iloc[ri_actual:ri_actual+n])
        IFm_nw=len(FuncObjN)-1
            #print(IF_minN[IFm_nw:IFm_nw+1])
        IFm_ant=len(FuncObj)-1
            #print(IF_min[IFm_ant:IFm_ant+1])
        rnd_actual=len(rnd)-n
            #print(rnd.iloc[rnd_actual:rnd_actual+n])

        for j in range(n):
            comparaA=float(ai.iat[ai_actual])
            comparar=float(ri.iat[ri_actual])
            comparaFN=float(FuncObjN[IFm_nw])
            comparaFB=float(FuncObj[IFm_ant])
            compararnd=float(rnd[rnd_actual])

            #print("ai")
            #print(ai)
            # If (rand > Ai and f(new_fitness < f(Best_fitness))
            # if all(rnd>ri) and all(FuncObj[1]< FuncObj[0])
            #print("aleatorio(",compararnd, ") < Sonoridad(", comparaA, ")")
            #if all((rnd.iloc[rnd_actual:rnd_actual+n]) <= (ri.iloc[ri_actual:ri_actual+n])):
            if compararnd <= comparaA:
                #print("                               SI ES MENOR-IF1")

                #print("     F.O.Nueva(",comparaFN, ") < F.O. Anterior(", comparaFB, ")")
                #if all(IF_minN[IFm_nw:IFm_nw+1] < FuncObj[IFm_ant:IFm_ant+1]):
                if comparaFN < comparaFB:    
                    #print("\n                               SI ES MENOR-IFinterno")
                    #print ("\n         Se reduce la sonoridad")
                    ai_new=alpha*(ai.iat[ai_actual])
                    #print("              Aantes", ai.iat[ai_actual],"Ahora",ai_new)
                    ai.iat[ai_actual]=ai_new

                    #print ("        Se incrementa la taza de pulso")
                    ri_new=ri.iat[ri_actual]*(1-e**(-gamma*1))
                    #print("                     Antes", ri.iat[ri_actual] , "Ahora", ri_new)
                    ri.iat[ri_actual]=ri_new

                #else:
                    #print("                                 NO ES MENOR-ELSE-interno")
                    #print("                                 No cambian los valores de sonoridad y taza de pulso")
                #print("------")
            #else:
                #print("\n                               NO ES MENOR-ELSE1")
                #print("Finaliza el ciclo", "\n")
                #print()

            ai_actual=ai_actual+1
            ri_actual=ri_actual+1
            rnd_actual=rnd_actual
        #print("no entre al IF")

        # iteración de incremento   
        it += 1

    #print(Resultados)



    print()
    print()
    print("**************************")
    print("Resultados Finales")
    print("**************************")
    #print(Resultados)

    print("  Resultados preliminares de TOPSIS:")
    print("                   ",ResultadosTOPSIS)
    # Mostrar los valores de la serie con una letra "A" agregada de manera horizontal
    #for value in ResultadosTOPSIS:
    #    print(f'A{value}', end=' ')
    #print("\n----------")
    print()

    print("  Resultados de cada iteración:")
    print("  ---------------------------------")
    print("   Iteración","  Mejor_alternativa")
    print("  ---------------------------------")
    for i in range(it):
        print("       ",i+1,"        ",Resultados[i])
    print("  ---------------------------------")




    #####################################################################################
    # Para almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()
    ejecut = hora_fin - hora_inicio
    tiempo_ejecucion = str(ejecut)
    

    # Imprimimos los resultados de tiempo
    print("Algoritmo TOPSIS-BA")
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:", ejecut)
    print()


    ####################################################################################
    ### Para guardar información en archivo de EXCEl

    base_filename = 'Experimentos/TOPSISBA'# Obtener el nombre del archivo base
    counter = 1 # Inicializar un contador para el nombre del archivo
    excel_filename = f'{base_filename}_{counter}.xlsx'

    ### --Verificar si el archivo ya existe, si es así, incrementar el contador
    while os.path.exists(excel_filename):
        counter += 1
        excel_filename = f'{base_filename}_{counter}.xlsx'


    ### -- Guardar los datos en un archivo xlsx
    dI={"alpha": [alpha], "gamma":[gamma], "No. de iteraciones":[iter_max]}
    dT= {"Algoritmo": ["TOPSIS-BA"],
        "Cantidad de Iteraciones": [iter_max ],
        "Hora de inicio": [hora_inicio.time()],
        "Fecha de inicio": [fecha_inicio],
        "Hora de finalización": [hora_fin.time()],
        "Tiempo de ejecución": [tiempo_ejecucion] }

    dataT = pd.DataFrame(dT)
    dataI = pd.DataFrame(dI)
    dataAlt = pd.DataFrame(RankFinNW)
    dataMOO=pd.DataFrame(ResultadosTOPSIS)
    dataResult = pd.DataFrame(Resultados)
    dataOrig=pd.DataFrame(raw_data)
    dataw = pd.DataFrame(w)
    dataBen=pd.DataFrame(benefit_attributes)
    dataNMx=pd.DataFrame(normalized_matrix)
    dataNMxW=pd.DataFrame(weighted_matrix)
    dataIb=pd.DataFrame(ideal_best)
    dataIw=pd.DataFrame(ideal_worst)
    dataSBt=pd.DataFrame(s_best)
    dataSwT=pd.DataFrame(s_worst)
    datapsc=pd.DataFrame(performance_score)
    alternativas = Resultados[-10:]

    datarii = pd.DataFrame(ri_ini)
    datari = pd.DataFrame(ri)
    dataaii =pd.DataFrame(ai_ini)
    dataai =pd.DataFrame(ai)
    dataf = pd.DataFrame(f)
    datarnd= pd.DataFrame(rnd)
    dataFOj =pd.DataFrame(FuncObj)
    dataFOjN =pd.DataFrame(FuncObjN)
    dataEV=pd.DataFrame(EV)
    datagbnw =pd.DataFrame(global_bestNW)
    datagb =pd.DataFrame(global_best)


    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        dataT.to_excel(writer, sheet_name='Tiempos')
        dataMOO.to_excel(writer, sheet_name='Resultados_TOPSIS')
        dataResult.to_excel(writer, sheet_name='Resultados')
        dataI.to_excel(writer, sheet_name='Valores_Iniciales')
        dataw.to_excel(writer, sheet_name='w')
        dataBen.to_excel(writer, sheet_name='Atributos_beneficios')
        dataEV.to_excel(writer, sheet_name='Eval_cardinales')
        dataOrig.to_excel(writer, sheet_name='Matriz_decisión')
        dataNMx.to_excel(writer, sheet_name='Matriz_normalizada')
        dataNMxW.to_excel(writer, sheet_name='Matriz_normalizada_xPesos')
        dataIb.to_excel(writer, sheet_name='Solución_ideal(SI)')
        dataIw.to_excel(writer, sheet_name='Solución_anti-ideal(SaI)')
        dataSBt.to_excel(writer, sheet_name='Distancia_SI')
        dataSwT.to_excel(writer, sheet_name='Distancia_SaI')
        datapsc.to_excel(writer, sheet_name='Puntuación_prox_relativa')
        dataAlt.to_excel(writer, sheet_name='Ranking_alternativas')
        datarii.to_excel(writer, sheet_name='Tasa_pulso(Inicial)')
        datari.to_excel(writer, sheet_name='Tasa_pulso(Final)')
        dataaii.to_excel(writer, sheet_name='Sonoridad(Inicial)')
        dataai.to_excel(writer, sheet_name='Sonoridad(Final)')
        dataf.to_excel(writer, sheet_name='Frecuencias')
        datarnd.to_excel(writer, sheet_name='Valores_Aleatorios')
        v.to_excel(writer, sheet_name='Velocidad')
        x.to_excel(writer, sheet_name='Posición')
        dataFOj.to_excel(writer, sheet_name='Función_Objetivo')
        dataFOjN.to_excel(writer, sheet_name='NW_Función_Objetivo')
        datagb.to_excel(writer, sheet_name='Mejor_global_local')
        datagbnw.to_excel(writer, sheet_name='NW_Mejor_global_local')

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
    dataMOO.to_csv(csv_filename, mode='a', index=False)
    dataResult.to_csv(csv_filename, mode='a', index=False)
    datagb.to_csv(csv_filename, mode='a', index=False)
    datagbnw.to_csv(csv_filename, mode='a', index=False)
    dataw.to_csv(csv_filename, mode='a', index=False)
    dataBen.to_csv(csv_filename, mode='a', index=False)
    dataEV.to_csv(csv_filename, mode='a', index=False)
    dataOrig.to_csv(csv_filename, mode='a', index=False)
    dataNMx.to_csv(csv_filename, mode='a', index=False)
    dataNMxW.to_csv(csv_filename, mode='a', index=False)
    dataIb.to_csv(csv_filename, mode='a', index=False)
    dataIw.to_csv(csv_filename, mode='a', index=False)
    dataSBt.to_csv(csv_filename, mode='a', index=False)
    dataSwT.to_csv(csv_filename, mode='a', index=False)
    datapsc.to_csv(csv_filename, mode='a', index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)
    datarii.to_csv(csv_filename, mode='a', index=False)
    datari.to_csv(csv_filename, mode='a', index=False)
    dataaii.to_csv(csv_filename, mode='a', index=False)
    dataai.to_csv(csv_filename, mode='a', index=False)
    dataf.to_csv(csv_filename, mode='a', index=False)
    datarnd.to_csv(csv_filename, mode='a', index=False)
    v.to_csv(csv_filename, mode='a', index=False)
    x.to_csv(csv_filename, mode='a', index=False)
    dataFOj.to_csv(csv_filename, mode='a', index=False)
    dataFOjN.to_csv(csv_filename, mode='a', index=False)
    dataAlt.to_csv(csv_filename, mode='a', index=False)
    print(f'Datos guardados en el archivo CSV: {csv_filename}')
    print()


    await asyncio.sleep(0.1)
    alternativas = [int(value) for value in alternativas]

    datosTopsisBa = {
        "mejor_alternativa": alternativas,
        "iteraciones": it,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }
    return datosTopsisBa