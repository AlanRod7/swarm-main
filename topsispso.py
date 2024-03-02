# Experimento TOPSIS
# https://www.kaggle.com/code/hungrybluedev/topsis-implementation
#
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# ACtualizado 06/Feb/2023

import numpy as np               # for linear algebra
import pandas as pd              # for tabular output
from scipy.stats import rankdata # for ranking the candidates
from ipaddress import v4_int_to_packed
from flask import Flask, render_template, request
from openpyxl import load_workbook
import random
from decimal import Decimal
import xlsxwriter
from cProfile import label
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import math
import random
from re import X
import asyncio
import datetime


############################################################
### Pre-requisites
# Los datos dados codificados en vectores y matrices
async def ejecutar_topsispso(w,wwi,c1,c2,T,r1,r2):
    
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión" )
    attributes = np.array(["C1", "C2", "C3", "C4", "C5"])
    #print("attributes",attributes)
    candidates = np.array(["A1", "A2", "A3", "A4", "A5", "A6","A7","A8","A9"])
    #print("candidates",candidates)
    n=5
    a=9
    Resultados=[]
    raw_data2=[]
    raw_data = np.array([
        [0.048, 0.047, 0.070, 0.087, 0.190],
        [0.053, 0.052, 0.066, 0.081, 0.058],
        [0.057, 0.057, 0.066, 0.076, 0.022],
        [0.062, 0.062, 0.063, 0.058, 0.007],
        [0.066, 0.066, 0.070, 0.085, 0.004],
        [0.070, 0.071, 0.066, 0.058, 0.003],
        [0.075, 0.075, 0.066, 0.047, 0.002],
        [0.079, 0.079, 0.066, 0.035, 0.002],
        [0.083, 0.083, 0.066, 0.051, 0.000],
    ])
    #print(raw_data)

    # Mostrar los datos sin procesar que tenemos
    pd.DataFrame(data=raw_data, index=candidates, columns=attributes)

    A1=pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    # Esta es la primera posición del enjambre
    CP=pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    # Solo para pruebas, ya que la primera velocidad es aleatoria
    #V=pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    #Es la primera mejor posición
    PBEST=pd.DataFrame(data=raw_data, index=candidates, columns=attributes)
    print(A1,"\n")


    print("\n -------------------------------------------")
    print("Controles iniciales" )
    print()
    print("Grado de preferencia para cada criterio")
    # Los pesos son por cada criterio
    #weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15]) #Original del artículo
    #w = np.array([0.3, 0.2, 0.2, 0.15, 0.15]) #Original del artículo

    #w=[float(0.200),float(0.200),float(0.200),float(0.200),float(0.200)]
    #weights = np.array([0.200, 0.200, 0.200, 0.200, 0.200])

    w=[float(0.400),float(0.200),float(0.030),float(0.070),float(0.300)]
    weights = np.array([0.400,0.200,0.030,0.070,0.300])

    #w=[float(0.123),float(0.099),float(0.043),float(0.343),float(0.392)]
    #weights = np.array([0.123, 0.099, 0.043, 0.343, 0.392])

    print(w,"\n")
    print("weights", w)
    # Los índices de los atributos (de base cero) que se consideran beneficiosos.
    # Se supone que los índices no mencionados son atributos de costos.
    benefit_attributes = set([0, 1, 2, 3, 4])

    #Aqui van los algoritmos /////////////////////////////////////////////////////////////////////////////
    wwi=0.7 # Tener un rango menor ayuda a
    c1=2.5    # Este influye en la pobabilidad hacia
    c2=2.5    # Este influye en la pobabilidad hacia
    dim=n*a #dimensión del enjambre
    T=3     #número de iteraciones para PSO
    rangoMin=0 #este rango de valores
    rangoMax=1  

    print("w(inertia) = ",wwi)
    print("c1 = ",c1)
    print("c2 = ",c2)
    print("No. de iteraciones = ",T)
    print("Rango de valores: (",rangoMin,",",rangoMax,") \n")
    #print("Objetive Fuction: Ri= (t^-i)/ (t^+i+t^-i) , i=1,...,m")

    ############################################################
    ###Step 1 - Normalizing the ratings
    ###PAso 1 - Normalizando las calificaciones
    m = len(raw_data) #Alternativas
    nn = len(attributes) #Criterios
    divisors = np.empty(n)
    for j in range(nn):
        column = raw_data[:,j]
        #print("column",column)
        divisors[j] = np.sqrt(column @ column)
        #print("divisors[",j,"]",divisors[j])
    raw_data /= divisors

    columns = ["$X_{%d}$" % j for j in range(nn)]
    pd.DataFrame(data=raw_data, index=candidates, columns=columns)
    #TEMPFO=pd.DataFrame(data=raw_data, index=candidates, columns=columns)
    #print("TEMPFO",TEMPFO)

    ############################################################
    ###Step 2 - Calculating the Weighted Normalized Ratings
    ###Paso 2 - Cálculo de las calificaciones normalizadas ponderadas
    raw_data *= weights
    pd.DataFrame(data=raw_data, index=candidates, columns=columns)

    ############################################################
    ###Step 3 - Identifying PIS ( A∗) and NIS ( A− )
    ###Paso 3 - Identificar
    a_pos = np.zeros(nn)
    a_neg = np.zeros(nn)
    for j in range(nn):
        column = raw_data[:,j]
        max_val = np.max(column)
        min_val = np.min(column)
        
        # See if we want to maximize benefit or minimize cost (for PIS)
        # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=columns)
    Ideal=pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=columns)
    #print("Ideal", Ideal)
    #print()
    r1 = pd.Series(a_pos)
    r2 = pd.Series(a_neg)
    #print("r1 \n",r1)
    #print("r2 \n",r2)
    #print()

    ############################################################
    ### Step 4 and 5 - Calculating Separation Measures and Similarities to PIS
    ### Pasos 4 y 5 - Cálculo de medidas de separación y similitudes con PIS
    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])


    pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["$S^*$", "$S^-$", "$C^*$"])
    SolPN=pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["$S^*$", "$S^-$", "$C^*$"])
    #print("SolPN",SolPN)
    #print()


    ############################################################
    ###Step 6 - Ranking the candidates/alternatives
    ###Paso 6 - Clasificación de las candidatas / alternativas
    def rank_according_to(data):
        ranks = rankdata(data).astype(int)
        ranks -= 1
        return candidates[ranks][::-1]

    cs_order = rank_according_to(cs)
    sp_order = rank_according_to(sp)
    sn_order = rank_according_to(sn)
    Alt=[]
    Alt=cs_order
    #print("ALTERNATIVAS", Alt)

    pd.DataFrame(data=zip(cs_order, sp_order, sn_order), index=range(1, m + 1), columns=["$C^*$", "$S^*$", "$S^-$"])


    ############################################################

    #print("La mejor alternativa es" + cs_order[0])
    #print("Las preferencias en orden descendente son " + ", ".j oin(cs_order) + ".")

    print("La mejor alternativa es: " + cs_order[0])
    print("La clasificación de las alternativas, de manera descendente es: " + ", ".join(cs_order) + ".")
    print()

    ############################################################
    ###PSO
    #####################

    # **********************************************************************PRIMERA ITERACIÓN
    print("\n ---------------------------------------------------------------------------------------")
    print(" ---------------------------------------------------------------------------------------")
    print("\n ITERACIÓN #1 -----------------------","\n")
    print("Rango de valores: (",rangoMin,",",rangoMax,") \n")
    print("w(inertia) = ",wwi)
    print("c1 = ",c1)
    print("c2 = ",c2)
    print("No. de iteraciones = ",T,"\n")
    print("r1 = ")
    print(r1,"\n")
    print("r2 = ")
    print(r2,"\n")
    #print("Objetive Fuction: Ri= (t^-i)/ (t^+i+t^-i) , i=1,...,m")


    # CURRENT VELOCITY (V)
    V = pd.DataFrame(columns=['C1','C2','C3','C4','C5'])
    for i in range(a):
        Vram1=[]
        for j in range(n):
            Vram=random.uniform(rangoMin,rangoMax)
            #print("Vram",Vram)
            Vram1.append(round((float(Vram)),3))
        Ver1 = pd.DataFrame({'C1':[Vram1[0]],'C2':[Vram1[1]],'C3':[Vram1[2]],'C4':[Vram1[3]],'C5':[Vram1[4]]})
        V = pd.concat([V,Ver1], ignore_index=True)
    print("V(1)=")
    print(V,"\n")


    # CURRENT POSITION (CP)
    #LA PRIMERA MEJOR POSICIÓN, SIEMPRE SERA LA PRIMERA POSICIÓN, NO SE TIENE ANTEDECENTES
    print("CP(1)=")
    print(CP,"\n")


    # FUNCIÓN OBJETIVO, CURRENT FITNESS (CF = Fx)
    #print("   Evaluar la función objetivo para obtener el mejor local y global.")
    #print("----------------------")
    #print("Objetive Fuction: Ri= (t^-i)/ (t^+i+t^-i) , i=1,...,m")
    #print("r1 \n",r1)
    #print("r2 \n",r2)
    #print()

    print("*********************************************")
    CF = pd.Series(r2)
    #print("CF")
    #print(CF)

    CFPS=[]
    for j in range(a):
        dat1=0
        for i in range(n):
            dat1= dat1+float(CP.iat[j,i])
        dat1=round((dat1/n),3)
        CFPS.append(float(dat1))
    Fx = pd.Series(CFPS)
    #print("LBF=")
    #print(Fx,"\n")


    # LOCAL BEST POSITION OF EACH PARTICLE UP TO FIRST ITERATION IS JUST ITS CURRENT POSITION
    # SINCE THER IS NO PREVIUO ITERATION EXISTS
    print("\n pbest(1)=")
    print(PBEST,"\n")


    # GLOBAL BEST FITNESS OF ITERATION #1
    GBF=[]
    pbestt=float(Fx.max())
    GBF.append(pbestt)
    print("GBF(1)=", GBF,"\n")


    # GLOBAL BEST POSITION OF ITERATION 1
    Fx_index=0
    for j in range(a):
        val1=round(float(GBF[0]),3)
        val2=round(float(Fx[j]),3)
        #print(val1,"=",val2)
        if(val1==val2):
            Fx_index=j
        #print("Fx_index",Fx_index)

    columna=[]
    for i in range(n):    
        Fx_P1=float(A1.iat[Fx_index,i])
        columna.append(Fx_P1)
        #print(Fx_P1)
    #print(columna)
    GBP = pd.DataFrame(columna)
    print("gbest(1)=")
    print(GBP)
    Resultados.append(Fx_index+1)

    print(" ---------------------------------------------------------------------------------------")
    print(" ---------------------------------------------------------------------------------------")
    print("           Mejor alternativa= A", Fx_index+1," para la iteración 1")
    print()



    """
    #GRAFICAR
    df=pd.read_csv('Graficar.csv')
    colors  = ("dodgerblue","salmon", "palevioletred", "steelblue", "seagreen", "plum", "blue", "indigo", "beige", "yellow")
    i=0

    for col in df:
        sizes=df[col].value_counts() #itera por cada una de las columnas y va contar la frecuencia de los valores
        pie=df[col].value_counts().plot(kind='pie', #tipo de gráfica
        colors=colors, #colores utilizados para la gráfica
        shadow=False, # este coloca sombra en la gráfica
        autopct='%1.1f%%', #formato del porcentaje con un decimal
        startangle=30, #es el angulo en el que inicia
        radius=1.5, # tiene que ver con las etiquetas que tiene en relacion con el centro
        center=(0.5,0.5), # este es donde estará el entro
        textprops={'fontsize':12}, #el tamaño de la letra
        frame=False, #si no quieren el recuadro
        labels=None, # este para que " plt.legend(labels..." funcione mejor
        pctdistance=0.75) #la distancia entre las etiquetas y los porcentajer
        labels=sizes.index.unique()  # etiquetas unicas de los datos
        plt.gca().axis("equal")  #para que los ejes se distribuyan de forma igual
        plt.title(df.columns[i],weight='bold',size=14)#para que cada título sea igual a el nombre de las columnas
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85) #este sirve para hacer el ajuste
        # PROBAR CANCELAR ESTO EN LA PIE
        plt.savefig(str(df.columns[i])+'.png',dpi=100, bbox_inches="tight") 
        #este indica el nombre de las gráficas como el de la columna el tight coloca un titulo se ajuste el título a la gráfica
        pie.set_ylabel('')# para que los títulos de la columna no esten an la parte izquierda de la gráfica, esto son los dataframes
        plt.legend(labels, bbox_to_anchor=(1,1)) # este es para hacer etiquetas en un cuadro fuera de la gráfica
        i=i+1
        plt.show()

    """


    ###############################################################################
    #ITERATION 2 a N
    t=1
    par=0
    longV1=0
    longseg=5
    iii=0
    columns = ["$X_{%d}$" % j for j in range(n)]
    TOPSISv1 = pd.DataFrame(columns=['C1','C2','C3','C4','C5'])
    TOPSISv2 = pd.DataFrame(columns=['C1','C2','C3','C4','C5'])

    while (t<T):
        print("ITERACIÓN #",t+1,"-----------------------","\n")
        print("Función Objetivo: Ri= (t^-i)/ (t^+i+t^-i) , i=1,...,m")
        print("Rango de valores: (",rangoMin,",",rangoMax,") \n")
        print("w(inertia) = ",wwi)
        print("c1 = ",c1)
        print("c2 = ",c2)
        print("No. de iteraciones = ",T,"\n")
        print("r1 (Actualizado) = ")
        print(r1.iloc[(len(r1)-n):len(r1)],"\n")
        print("r2 (Actualizado) = ")
        print(r2.iloc[(len(r2)-n):len(r2)],"\n")

        Fxce=[]
        ii=0
        longVel=a*t
        #print("tr12",tr12)
        for j in range(a):
            otroV=[]
            otroCP=[]
            tr12=(len(r1)-n)
            CAA=(len(CP)-a)
            GBP12=(len(GBP)-n)
            #print("                                                  CAA",CAA)
            #print("                                                  tr12",tr12)
            #print("                                                  GBP12",GBP12)
            for i in range(n):
                #print("                                             j,i ", j ,i)
                # 1-a) ACTUALIZANDO LA VELOCIDAD            
                #Vtt1=0
                Vtt1=float(V.iat[CAA,i])
                Vt11=float((wwi*Vtt1))
                #print("Vt11",round((Vt11),3))

                PBESTtt=float((PBEST.iat[CAA,i]))
                rr1=float(r1.iat[tr12])
                CPtt=float((CP.iat[CAA,i])) 
                Vt12=float((c1*rr1*(PBESTtt-CPtt)))
                #print("c1",c1)
                #print("rr1",rr1)
                #print("PBESTtt",PBESTtt)
                #print("CPtt",CPtt)
                #print("Vt12",round((Vt12),3))
                
                GBPtt=float(GBP.loc[GBP12])
                rr2=float(r2.iat[tr12])
                #print("c2",c2)
                #print("rr2",rr2)
                #print("GBPtt",GBPtt)
                #print("CPtt",CPtt)
                Vt13=float((c2*rr2*(GBPtt-CPtt)))          
                #print("Vt13",round((Vt13),3))
                #print("--- \n")

                VFn=round((float(Vt11+Vt12+Vt13)),3)
                #print("VFn",round((VFn),3))
                otroV.append(float(VFn))

                # 2-a) ACTUALIZANDO LA PRIMERA  POSICIÓN
                CPtt2=float((CP.iat[CAA,i])) 
                #print(CPtt2)
                #print(VFn)
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
                #print("OtroCP")
                #print(otroCP)
                tr12=tr12+1
                GBP12=GBP12+1

            V.loc[len(V.index)]=otroV
            CP.loc[len(CP.index)]=otroCP
            CAA=CAA+1
            #print("OtroCP")
            #print(otroCP)
            raw_data2 = np.append(raw_data2,otroCP)  
            #print(raw_data2)
            #print("------ \n")
        #print(V)
        #print(CP)
        #print("raw_data2")
        #print(raw_data2)
        #print()


        # --- Step 1 - Normalizing the ratings 
        # --- Paso 1 - Normalizando las calificaciones
        
        divisors = np.empty(n)
        #print("divisors",divisors)
        ifo=0
        for j in range(a):
            TPS1=[]
            for i in range(n):
                #print("divisors[i]",divisors[i])
                column = raw_data2[ifo]
                #print("column",column)
                ifo=ifo+1
                raw_dataT3 = column/divisors[i]
                #raw_data3 = np.append(raw_data3,raw_dataT3) 
                TPS1.append(round((float(raw_dataT3)),3))
            
            TPSVer1 = pd.DataFrame({'C1':[TPS1[0]],'C2':[TPS1[1]],'C3':[TPS1[2]],'C4':[TPS1[3]],'C5':[TPS1[4]]})
            TOPSISv1 = pd.concat([TOPSISv1,TPSVer1], ignore_index=True)
        #print("[X]=")
        #print(TOPSISv1)

        # --- Step 2 - Calculating the Weighted Normalized Ratings
        # --- Paso 2 - Cálculo de las calificaciones normalizadas ponderadas
        # print("weights",weights)
        for j in range(a):
            TPS3=[]
            for i in range(n):
                TPS2=float(TOPSISv1.iat[j,i])
                #print("TPS2",TPS2)
                pessos=weights[i]
                #print("pessos",pessos)
                raw_dataT5 = TPS2*pessos
                #print("[x]=",TPS2,"x",pessos,"=",raw_dataT5)
                TPS3.append(round((float(raw_dataT5)),3))
            
            TPSVer2 = pd.DataFrame({'C1':[TPS3[0]],'C2':[TPS3[1]],'C3':[TPS3[2]],'C4':[TPS3[3]],'C5':[TPS3[4]]})
            TOPSISv2 = pd.concat([TOPSISv2,TPSVer2], ignore_index=True)
        #print("[Y]=")
        #print(TOPSISv2)


        # --- Step 3 - Identifying PIS ( A∗) and NIS ( A− )
        # --- Paso 3 - Identificar
        a_pos = np.zeros(n)
        a_neg = np.zeros(n)
        for j in range(n):
            TPS6=[]
            for i in range(a):
                TPS5=float(TOPSISv2.iat[i,j])
                #print(TPS5)
                TPS6.append(round((float(TPS5)),3))

            #print(TPS6)
            max_val = np.max(TPS6)
            min_val = np.min(TPS6)
            #print("max_val, min_val")
            #print(max_val, min_val)
            #print("---")

            # See if we want to maximize benefit or minimize cost (for PIS)
            # Ver si queremos maximizar el beneficio o minimizar el costo (para PIS)
            if j in benefit_attributes:
                a_pos[j] = max_val
                a_neg[j] = min_val
            else:
                a_pos[j] = min_val
                a_neg[j] = max_val

            #print("NEG=",a_neg[j])
            #print("POS=",a_pos[j])
            #print("-----")

            pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=columns)
            Ideal=pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=columns)
            #print("Ideal", Ideal)
            #print()
            r1TPS = pd.Series(a_pos)
            r2TPS = pd.Series(a_neg)
        
        r1S = pd.Series(r1TPS)
        r1 = pd.concat([r1,r1S], ignore_index=True)
        r2S = pd.Series(r2TPS)
        r2 = pd.concat([r2,r2S], ignore_index=True)
        #print("r1-Actualizado")
        #print(r1)
        #print("r2-Actualizado")
        #print(r2)
        #print(r2S)
        #print()

        CFxe = pd.Series(r2TPS)
        CF = pd.concat([CF,CFxe], ignore_index=True)
        #print("CF=")
        #print(CF,"\n")

        # LOCAL BEST POSITION OF EACH PARTICLE UP TO FIRST ITERATION IS JUST ITS CURRENT POSITION
        # SINCE THER IS NO PREVIUO ITERATION EXISTS
        CFPS=[]
        altFXx=(t*a)
        for j in range(a):
            dat1=0
            for i in range(n):
                #print("           ",float(CP.iat[altFXx,i]))
                dat1= dat1+float(CP.iat[altFXx,i])
                #print("                    dat1",dat1)
            altFXx=altFXx+1
            dat1=round((dat1/5),3) # entre 5 porque son 5 partículas
            #print("FINAL",dat1,"\n")
            CFPS.append(float(dat1))
        Fx12 = pd.Series(CFPS)
        Fx = pd.concat([Fx,Fx12], ignore_index=True)
        #print("LBF=")
        #print(Fx,"\n")

        zz1=0
        z1=0
        
        if t==1:
            cont_act=n
            cont_ant=0
        else:
            cont_act=longseg-n
            cont_ant=cont_act-n
        
        for j in range(n):
            longsegP=len(CP)-a
            #print("                                  longsegP-INICIAL,z1",longsegP,z1)#18 (CP ACTUAL)
            
            #print(cont_ant,cont_act)
            actual=float(CF.iat[cont_act])
            anterior=float(CF.iat[cont_ant])
            LxCP=[]

            #print(actual,">" ,anterior)
            if (actual>anterior) or (actual==anterior): #CP(2)
                #print("entre al IF")
                for z in range(a):
                    #print("                                  longsegP-USADO,i",longsegP,z1)#9
                    x1=CP.iat[longsegP,z1]
                    LxCP.append(round((x1),3))
                    #print("         CP ACtual",round((x1),3))
                    longsegP=longsegP+1
            else:  # CP(1)
                #print("entre al ELSE")
                for z in range(a):
                    longsegPt=longsegP-a
                    #print("                                  longsegP-USADO",longsegP)#9
                    #print("                                     longsegPt-USADO,i",longsegPt,z1)#9
                    x1=CP.iat[longsegPt,z1]
                    LxCP.append(round((x1),3))
                    #print("CP anterior",round((x1),3))
                    longsegP=longsegP+1
                        
            z1=z1+1
            cont_ant=cont_ant+1
            cont_act=cont_act+1 
            #print(LxCP)
            if j==0:
                Cc1 = LxCP
                #print("Guarde eb cc1")
            if j==1:
                Cc2 =LxCP
                #print("Guarde eb cc2")
            if j==2:
                Cc3=LxCP
                #print("Guarde eb cc3")
            if j==3:
                Cc4=LxCP
                #print("Guarde eb cc4")
            if j==4:
                Cc5=LxCP
                #print("Guarde eb cc5")
        new_CPLxCont = pd.DataFrame()
        new_CPLxCont['C1']=Cc1
        new_CPLxCont['C2']=Cc2
        new_CPLxCont['C3']=Cc3
        new_CPLxCont['C4']=Cc4
        new_CPLxCont['C5']=Cc5
        #print("new_CPLxCont")
        #print(new_CPLxCont)
        PBEST = pd.concat([PBEST,new_CPLxCont], ignore_index=True)
        #print("PBEST")       
        #print(PBEST)
        #print()


        # GLOBAL BEST FITNESS OF ITERATION 
        #print("Fx")
        #print(Fx12)
        pbestt2=float(max(Fx12))
        GBF.append(pbestt2)

    
        # GLOBAL BEST POSITION OF ITERATION 
        Fx_index=0
        temp_GBP=len(Fx)-a
        for j in range(a):
            val1=round(float(GBF[t]),3)
            val2=round(float(Fx[temp_GBP]),3)
            #print(val1,"=",val2)
            if(val1==val2):
                Fx_index=j
            temp_GBP=temp_GBP+1
        Fx_index=Fx_index+(a*t)
        
        #print(CP)
        #print("++++Entre")
        columna=[]
        for i in range(n):    
            #print("Fx_index,i",Fx_index,i)
            Fx_P1=float(CP.iat[Fx_index,i])
            columna.append(Fx_P1)
            #print(Fx_P1)
        #print(columna)
        GBP12 = pd.Series(columna)
        GBP = pd.concat([GBP,GBP12], ignore_index=True)
        #print("gbest(1)=")
        #print(GBP,"\n")
        Fx_index=Fx_index-(a*t)
        #print(Fx_index)
        Resultados.append(Fx_index+1)
        #print("           Mejor alternativa= A", Fx_index+1," para la iteración 1")
        #print("      ---------------------------------------------------------------")
        #print()


        # IMPRESIÓN DE RESULTADOS  
        seg=a*t #5*1=5
        print("V(",t+1,") =")
        print(V.iloc[seg:seg+a,:],"\n")
        print("CP(",t+1,") =")
        print(CP.iloc[seg:seg+a,:],"\n")
        print("pbest(",t+1,") =")
        print(PBEST.iloc[seg:seg+a],"\n")
        print("GBF =", GBF[t],"\n")
        print("gbest(",t+1,") =")
        print(GBP.iloc[(len(GBP)-n):len(GBP)],"\n")
        #Mejor=(Fx_index2+1)-(a*t) si fuera CP
        Mejor=(Fx_index+1)
        print("           Mejor alternativa= A", Mejor," para la iteración",t+1)
        print("      ---------------------------------------------------------------")
        #Resultados.append(Mejor)
        ii=ii+1
        iii=iii+1
        t=t+1


    print()
    print()
    print("**************************")
    print("Resultados Finales")
    print("**************************")
    #print("   Mejor posición=")
    #print(GBP.iloc[(len(GBP)-n):len(GBP)],"\n")
    #print("   Mejor óptimo=", GBF[t-1], "\n")


    print("   Ranking_TOPSIS= ",Alt[0])
    print("   Ranking_alternativas=",Alt)
    print()
    print("   Iteración","  Mejor_alternativa")
    print("  ---------------------------------")
    dd=0
    for i in range(T):
        print("       ",i+1,"        ","A",Resultados[i])
    print("  ---------------------------------")

    dI={"w(inertia)":wwi, "c1":c1, "c2":c2, "No. de iteraciones":T, "Función Objetivo": ['IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}'],"Función_objetivoPSO:":['Min f(x_{1},x_ {2}) =(x_{1}^{2} + (x_{2}\)^{2}")'], "Rango_Min":rangoMin,"Rango_Max":rangoMax}
    dataI = pd.DataFrame(dI)
    dataGBF = pd.DataFrame(GBF)
    dataGBP = pd.DataFrame(GBP)
    dataAlt = pd.DataFrame(Alt)
    dataw = pd.DataFrame(w)
    dataResult = pd.DataFrame(Resultados)
    alternativas = Resultados[-10:]
    hora_fin = datetime.datetime.now()


    with pd.ExcelWriter('Experimentos2/TOPSISPSO.xlsx', engine='xlsxwriter') as writer:
        dataI.to_excel(writer, sheet_name='Iniciales')
        r1.to_excel(writer, sheet_name='r1')
        r2.to_excel(writer, sheet_name='r2')
        dataw.to_excel(writer, sheet_name='w')
        A1.to_excel(writer, sheet_name='Matriz')
        dataAlt.to_excel(writer, sheet_name='Ranking_alternativas')
        V.to_excel(writer, sheet_name='Velocity')
        CP.to_excel(writer, sheet_name='Position')
        PBEST.to_excel(writer, sheet_name='PBEST')
        Fx.to_excel(writer, sheet_name='Fx')
        dataGBF.to_excel(writer, sheet_name='GBF')
        dataGBP.to_excel(writer, sheet_name='gbest')
        dataResult.to_excel(writer, sheet_name='Resultados')

    print('Datos guardados el archivo:TOPSISPSO.xlsx')
    print()
    #Imprimimos los resultados de tiempo
    print("Algoritmo TOPSISPSO")
    print("Cantidad de Iteraciones:", t)
    print("Hora de inicio:", hora_inicio.time())
    print("Fecha de inicio:",fecha_inicio)
    print("Hora de finalizacion:", hora_fin.time())
    print("Tiempo de ejecucion:", hora_fin-hora_inicio)
    print("")
    
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