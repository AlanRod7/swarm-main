
#
# BAT ALGORITHM
# 
import pandas as pd
import numpy as np
from math import e
import math
import random
import datetime
import asyncio
async def ejecutar_ba(w, wwi, c1, c2, T, r1, r2):
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()

######################################################################################################################################################
# Paso # 1: Inicialice la primera posición y velocidad
#
#
    n=9 #Alternativas
    d=5 # Criterios
#print (n) #a=9   # Alternativas
#print (d) #n=6   # Criterior
#n = x.shape[0]
#d = x.shape[1]

# ----Iteraciones
#iter_max = int(input("\n""Ingrese el numero de iteraciones maximas: \n"))
    iter_max = 10

#-------------- Posición inicial
#               lugar donde algo se encuentra, es decir la posición de los mirciélagos
#
#- comando para ingresar desde archivo de Excel
#x0= pd.read_csv('Pos_Inicial.cvs')
    A1t={'C1':[0.048,0.053,0.057,0.062,0.066,0.070,0.075,0.079,0.083],
        'C2':[0.047,0.052,0.057,0.062,0.066,0.071,0.075,0.079,0.083],
        'C3':[0.070,0.066,0.066,0.063,0.070,0.066,0.066,0.066,0.066],
        'C4':[0.087,0.081,0.076,0.058,0.085,0.058,0.047,0.035,0.051],
        'C5':[0.190,0.058,0.022,0.007,0.004,0.003,0.002,0.002,0.000]}
    x=pd.DataFrame(A1t)
    print("Posición inicial=")
    print(x,"\n")

#-------------- Velocidad inicial
#           velocidad de algo a una dirección dada para cada murciélago
#
#- Comando para ingresar datos desde excel
#v0= pd.read_csv('Vel_Inicial.cvs')
#v0 = pd.DataFrame(columns=['C1','C2','C3','C4','C5'])
#for i in range(d):
#    Vram1=[]
#    for j in range(n):
#        Vram1.append(0.00)
#    Ver1 = pd.DataFrame({'C1':[Vram1[0]],'C2':[Vram1[1]],'C3':[Vram1[2]],'C4':[Vram1[3]],'C5':[Vram1[4]]})
#    v0 = pd.concat([v,Ver1], ignore_index=True)

# valores de cero para la velocidad
#v =pd.DataFrame(0, index=range(n), columns=range(d))
#v.columns = ['C1','C2','C3','C4','C5']
#Velocidad con valores aleaorios
    v = pd.DataFrame(columns=['C1','C2','C3','C4','C5'])
    for i in range(n):
        Vram1=[]
        for j in range(d):
            Vram=random.uniform(0,1)
            #print("Vram",Vram)
            Vram1.append(round((float(Vram)),3))
        Ver1 = pd.DataFrame({'C1':[Vram1[0]],'C2':[Vram1[1]],'C3':[Vram1[2]],'C4':[Vram1[3]],'C5':[Vram1[4]]})
        v = pd.concat([v,Ver1], ignore_index=True)
    print("Velocidad inicial=")
    print(v,"\n")


#-------------- Configuración de parámetros
#
    alpha=0.90   # Valor para actualizar Ai Loudness 
    gamma=0.90   # Valor para actualizar ri Pulse Rate

# ----Tasa de pulso (Pulse rate) 
#Valores para tasa de pulso [0,1]
#print("Ingrese los valores iniciales para ri / Pulse emission: [0,1]")
    ri=[]  
#for a in range (n):
     #ri0 = float(input("BAT" + " " + str(a+1) + "\n"))
#     ri0=0.1
#     ri.append(ri0)
#ri=pd.Series(ri)
    ri = pd.Series(np.random.rand(n))
    ri_ini=ri
#print(ri)

# ----Sonoridad (Loudness)
#         Característica de sonido y significa cuán fuerte o suave es el sonido al oyente y denotaremos la sonoridad con Ai
#         Valores sonoridad [1,2]
#print("\n""Ingrese los valores iniciales para la sonoridad Ai / Loudness: [1,2]")
    ai=[]   
#for b in range (n):
     #ai0 = float(input("BAT" + " " + str(b+1) + "\n"))
#     ai0 = 0.95
#     ai.append(ai0)
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

    print("Configuración de parámetros:")
    print("             alpha",alpha) 
    print("             gamma",gamma)
    print("             Número de iteraciones:",iter_max )
    print("\n Tasa de pulso (Pulse rate)")
    print(ri)
    print("\n Sonoridad (Loudness)")
    print(ai)
    print("\n Frecuencia")
    print(f)
    print("\n Rango de frecuencia: [",fmin,",",fmax,"]")
    print("\n Valores Aleatorios")
    print(rnd)
    print("--------------------------------------------------  \n")


######################################################################################################################################################
################################# Primera iteracion

# Iteración
    it = 0
    Resultados = []

    while it < iter_max:
        print('\n =======================================================')
        print("ITERACIÓN #",it)

        ######################################################
        # Calcular el valor de la función objetivo en cada x:  1+2x-x^2
        #
    #
        fitness =pd.DataFrame(columns=['C1','C2','C3','C4','C5'])
        FuncObj=[]
        IF_max=[]
        IF=[] #Initial_Fitnes

        x_actual=len(x)-n
        #print()
        #print("posición")
        #print(x)
        for j in range(n):
            P1=0
            P2=0
            IFtt=[]
            for i in range(d):
                #print("x.iat[i,j]",x.iat[j,i])
                #print("x.iat[i,j]",x.iat[x_actual,i])
                #P1 = float(x.iat[x_actual,i]**2)
                P1 = float(x.iat[x_actual,i])
                #print("P1",P1)
                #P2 = P2 + P1
                P2 = round(float(1+(2*P1)-(P1*P1)),3)
                #print("    P2",P2)
                IFtt.append(round((float(P2)),3))
                #print("\n Fitness",IFtt)

            x_actual=x_actual+1
            fitness_min = (round((float(min(IFtt))),3))
            FuncObj.append(round((float(fitness_min)),3))
            #print("\n fitness_min= ",FuncObj)

            rang1 = pd.DataFrame({'C1':[IFtt[0]],'C2':[IFtt[1]],'C3':[IFtt[2]],'C4':[IFtt[3]],'C5':[IFtt[4]]})
            fitness = pd.concat([fitness,rang1], ignore_index=True)

        IF_max.append(round((float(min(FuncObj))),3))
        IF_maxt=(round((float(min(FuncObj))),3))
        print("\n Función objetivo= ",FuncObj)
        print("\n Función objetivo(min)= ",IF_max)
        #print("\n fitness_max= ",IF_maxt)
        print("fitness \n",fitness)
        print()   


        global_best =pd.DataFrame(columns=['C1','C2','C3','C4','C5'])
        for j in range(n):
            if FuncObj[j]==IF_maxt:

                Mejor=j # guarda la mejor alternativa
                #print("La mejor alternativa es A", Mejor)
                CBt=[]
                for z in range(d):

                    CB1= float(x.iat[j,z])
                    #print("CB1",CB1)
                    CBt.append(CB1)
                    #print("CBt",CBt)
            #print("CBt",CBt)
        #print("La mejor alternativa es A", Mejor+1)
        CCB = pd.DataFrame({'C1':[CBt[0]],'C2':[CBt[1]],'C3':[CBt[2]],'C4':[CBt[3]],'C5':[CBt[4]]})
        #print("CCB",CCB)
        global_best = pd.concat([global_best,CCB], ignore_index=True)
        print("El mejor global local: ")
        print(global_best, "\n")
        #print("------------------------------------------------------")

        #Result.append(Mejor+1)
        Resultados.append(Mejor+1)


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

        for i in range(n):#n alternativas
            VAct=[]
            f_actual=len(f)-1
            global_actual=len(global_best)-1

            for j in range(d): #dcriterios

                # Vi = Velocidad inicial + (Posición actual - Mejor_posición) * Frecuencia
                VAct1 = v.iat[v_actual,j] + (x.iat[x_actual,j] - global_best.iat[global_actual,j]) * f.iat[f_actual,j]
                VAct.append(round((float(VAct1)),3))

                #print("     x.iat[i,j]",x.iat[x_actual,j])
                #print("     v.iat[i,j]",v.iat[v_actual,j])
                #print("     f.iat[f_actual,i]",f.iat[f_actual,j])
                #print("     global_best.iat[0,i]",global_best.iat[global_actual,j])

                #print(v.iat[j,v_actual],"+ (", x.iat[j,x_actual], "-", global_best.iat[global_actual,i], ")*", f.iat[f_actual,i])
                #print("   VAct===", VAct1)
                #print("---")

            x_actual=x_actual+1
            v_actual=v_actual+1
            #print("v_new",v_actual)
            Ver1 = pd.DataFrame({'C1':[VAct[0]],'C2':[VAct[1]],'C3':[VAct[2]],'C4':[VAct[3]],'C5':[VAct[4]]})
            v = pd.concat([v,Ver1], ignore_index=True)
        print("-----")
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

            x_actual=x_actual+1
            v_actual=v_actual+1

            Ver2 = pd.DataFrame({'C1':[XAct[0]],'C2':[XAct[1]],'C3':[XAct[2]],'C4':[XAct[3]],'C5':[XAct[4]]})
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
                    rrtP7 = float(global_best.iat[0,j])
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
                x_actual=x_actual+1
                v_actual=v_actual+1
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

        # Nueva función objetivo:  suma (x_i^2)
        fitnessN =pd.DataFrame()
        FuncObjN=[]
        IF_maxN=[]
        IFN=[] #Initial_Fitnes
        x_actual=len(x)-n

        for j in range(n):
            P1=0
            P2=0
            IFtt=[]
            for i in range(d):
                #print("x.iat[i,j]",x.iat[j,i])
                #print("x.iat[i,j]",x.iat[x_actual,i])
                #P1 = float(x.iat[x_actual,i]**2)
                P1 = float(x.iat[x_actual,i])
                #print("P1",P1)
                #P2 = P2 + P1
                P2 = round(float(1+(2*P1)-(P1*P1)),3)
                #print("    P2",P2)
                IFtt.append(round((float(P2)),3))
                #print("\n Fitness",IFtt)

            x_actual=x_actual+1
            fitness_minN = min(IFtt) 
            FuncObjN.append(round((float(fitness_minN)),3))
            #print("\n fitness_min= ",IF_minN)

            rang1N = pd.DataFrame({'C1':[IFtt[0]],'C2':[IFtt[1]],'C3':[IFtt[2]],'C4':[IFtt[3]],'C5':[IFtt[4]]})
            fitnessN = pd.concat([fitnessN,rang1N], ignore_index=True)
            #print("----------")

        IF_maxNt= min(FuncObjN)
        IF_maxN.append(round((float(min(FuncObjN)))))
        #print("\n fitness_min= ",IF_minN)
        #print("\n fitness_max= ",IF_maxN)
        #print("fitness \n",fitness)
        #print()   

        print("\n fitness_min= ",IF_maxt)
        #print("fitness \n",fitness)
        print(" nUEVO_fitness_min= ",IF_maxNt)
        #print("nUEVO_fitness \n",fitnessN)
        print()   


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

    print("   Iteración","  Mejor_alternativa")
    print("  ---------------------------------")
    for i in range(it):
        print("       ",i+1,"        ","A",Resultados[i])
    print("  ---------------------------------")

    # PAra almacenar tiempo de ejecución
    hora_fin = datetime.datetime.now()

    # PAra guardar información en archivo de EXCEl
    dI={"alpha":alpha, "gamma":gamma, "No. de iteraciones":iter_max, "Función_objetivoPSO:":['Min f(x_{1},x_ {2}) =(x_{1}^{2} + (x_{2}\)^{2}")'], "Rango_Frecuencia(min)":fmin,"Rango_Frecuencia(max)":fmax}
    #General={"Algoritmo BA clásico":0, "Cantidad de Iteraciones":iter_max, "Hora de inicio": hora_inicio.time(), "Fecha de inicio": fecha_inicio, "Hora de finalización": hora_fin.time(), "Tiempo de ejecución":hora_fin-hora_inicio}

    #dataGral =pd.DataFrame(General)
    dataI = pd.DataFrame(dI)
    dataResult = pd.DataFrame(Resultados)
    datarii = pd.DataFrame(ri_ini)
    datari = pd.DataFrame(ri)
    dataaii =pd.DataFrame(ai_ini)
    dataai =pd.DataFrame(ai)
    dataf = pd.DataFrame(f)
    datarnd= pd.DataFrame(rnd)
    #dataFmin= pd.DataFrame(FuncObj)
    #dataFmax= pd.DataFrame(IF_max)
    alternativas = Resultados[-10:]
    

    with pd.ExcelWriter('Experimentos2/BA.xlsx', engine='xlsxwriter') as writer:
        #dataGral.to_excel(writer, sheet_name='Generales')
        dataResult.to_excel(writer, sheet_name='Resultados')
        dataI.to_excel(writer, sheet_name='Iniciales')
        datarii.to_excel(writer, sheet_name='Tasa_pulso(Inicial)')
        datari.to_excel(writer, sheet_name='Tasa_pulso(Final)')
        dataaii.to_excel(writer, sheet_name='Sonoridad(Inicial)')
        dataai.to_excel(writer, sheet_name='Sonoridad(Final)')
        dataf.to_excel(writer, sheet_name='Frecuencia')
        datarnd.to_excel(writer, sheet_name='No_Aleatorios')
        #dataFmin.to_excel(writer, sheet_name="Función objetivo")
        #dataFmax.to_excel(writer, sheet_name="Función objetivo(valor)")
        v.to_excel(writer, sheet_name='Velocidad')
        x.to_excel(writer, sheet_name='Posición')
        #global_best.to_excel(writer, sheet_name='PBEST')

    print('Datos guardados el archivo:BA.xlsx')
    print()

    # Imprimimos los resultados de tiempo
    print("Algoritmo BA clásico")
    print("Cantidad de Iteraciones:", it)
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:", fecha_inicio)
    print("Hora de finalización:", hora_fin.time())
    print("Tiempo de ejecución:",hora_fin-hora_inicio)
    print()

    await asyncio.sleep(0.1)
    
    datosBa = {
        "mejor_alternativa": alternativas,
        "iteraciones": it,
        "hora_inicio": hora_inicio.time().strftime('%H:%M:%S'),
        "fecha_inicio": fecha_inicio.isoformat(),
        "hora_finalizacion": hora_fin.time().strftime('%H:%M:%S'),
        "tiempo_ejecucion": str(hora_fin - hora_inicio)
    }
    
    return datosBa

