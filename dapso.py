from flask import Flask, render_template, request
from ipaddress import v4_int_to_packed
from flask import Flask, render_template, request
from openpyxl import load_workbook
import random
from decimal import Decimal
import pandas as pd
import numpy as np
import xlsxwriter
from cProfile import label
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import datetime
import asyncio


async def ejecutar_dapso(w, wwi, c1, c2, T, r1, r2):
    
    hora_inicio = datetime.datetime.now()
    fecha_inicio = hora_inicio.date()
    print()
    print("-------------------------------------------")
    print("Construcción de la matriz de decisión")
    candidates = np.array(["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"])
    n = 5
    a = 9
    Resultados = []
    A1t = {'C1': [0.048, 0.053, 0.057, 0.062, 0.066, 0.070, 0.075, 0.079, 0.083],
           'C2': [0.047, 0.052, 0.057, 0.062, 0.066, 0.071, 0.075, 0.079, 0.083],
           'C3': [0.070, 0.066, 0.066, 0.063, 0.070, 0.066, 0.066, 0.066, 0.066],
           'C4': [0.087, 0.081, 0.076, 0.058, 0.085, 0.058, 0.047, 0.035, 0.051],
           'C5': [0.190, 0.058, 0.022, 0.007, 0.004, 0.003, 0.002, 0.002, 0.000]}
    A1 = pd.DataFrame(data=A1t, index=candidates)
    CP = pd.DataFrame(A1t)  # Esta es la primera posición del enjambre
    # V=pd.DataFrame(A1t) # Solo para pruebas
    PBEST = pd.DataFrame(A1t)  # Es la primera mejor posición
    print(A1, "\n")

    print("\n -------------------------------------------")
    print("Controles iniciales")

    print("-------------------------------------------")
    print("Grado de preferencia para cada alternativa")
    # Los pesos son por cada criterio
    # w=[float(0.200),float(0.200),float(0.200),float(0.200),float(0.200)]
    #w = [float(0.400), float(0.200), float(0.030), float(0.070), float(0.300)]
    # w=[float(0.123),float(0.099),float(0.043),float(0.343),float(0.392)]
    print(w, "\n")

    #wwi = 0.7  # Tener un rango menor ayuda a
    #c1 = 2.5    # Este influye en la pobabilidad hacia
    #c2 = 2.5    # Este influye en la pobabilidad hacia
    dim = n*a  # dimensión del enjambre
    #T = 5  # número de iteraciones para PSO
    rangoMin = 0  # este rango de valores
    rangoMax = 1

    print("w(inertia) = ", wwi)
    print("c1 = ", c1)
    print("c2 = ", c2)
    print("No. de iteraciones = ", T, "\n")
    # print("r1 = ")
    # print(r1,"\n")
    # print("r2 = ")
    # print(r2,"\n")
    print(
        "Función Objetivo: IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}")
    print("Rango de valores: (", rangoMin, ",", rangoMax, ") \n")

    # a)Solución ideal
    St = []
    print("-------------------------------------------")
    print("Establecer la solución ideal")
# Mejor solución del conjunto de los datos.
# 1) promedio de cada criterio

    for j in range(n):
        P1 = 0
        for i in range(a):
            P1 = float(P1)+float(A1.iat[i, j])
        P1 = round((float(P1)/float(a)), 3)
        St.append(round((float(P1)), 3))
    S = pd.Series(St)
    r1 = pd.Series(St)
    print(S, "\n")

    print("-------------------------------------------")
    print("Determinar el índice de similitud")
    # a) normalizamos (a/S)
    # b) elevar lo normalizado al peso(w)
    # c) Producto sucesivo
    CFt = []
    SI1 = []
    PST = []

    ISSFO = pd.DataFrame(columns=['C1', 'C2', 'C3', 'C4', 'C5'])
    for j in range(a):
        SI1 = []
        ISSFOm1 = []
        for i in range(n):
            dat1 = float(A1.iat[j, i])
            dat2 = float(S[i])
            wn2 = float(w[i])
            dat3 = round((dat1/dat2), 3)
            # print("normalizado",dat3)

            # print(dat1,"/",dat2, "=",dat3)
            dat4 = round((abs(dat3)**abs(wn2)), 3)
            # print("Elevado",dat4)
            # print("                         ",dat3,"^",wn2,"=",dat4)
            # SI1.append(float(dat4))
            ISSFOm1.append(round((float(dat4)), 3))
            # print()
        # print("SI1")
        # print(SI1)
        ISSFOVr1 = pd.DataFrame({'C1': [ISSFOm1[0]], 'C2': [ISSFOm1[1]], 'C3': [
                                ISSFOm1[2]], 'C4': [ISSFOm1[3]], 'C5': [ISSFOm1[4]]})
        ISSFO = pd.concat([ISSFO, ISSFOVr1], ignore_index=True)
    # print("ISSFO(1)=")
    # print(ISSFO,"\n")

    for j in range(a):
        Sqq1 = float(1)
        for z in range(n):
            dat5 = float(ISSFO.iat[j, z])
            # print("valor",dat5)
            # Sqq1=(float(Sqq1)*float(SI1[z]))
            Sqq1 = (Sqq1*dat5)
            # print("              Sqq1",Sqq1)
            # print("              dat5",dat5)
        # print("-----------")
        Sqq1 = round(Sqq1, 3)
        # print("producto",Sqq1)
        # print()
        CFt.append(float(Sqq1))
        PST.append(float(Sqq1))
    PSS = pd.Series(CFt)
    r2 = pd.Series(CFt)
    # print(r2)
    print("-- Índice de similitud =")
    print(PSS)

    print("\n -------------------------------------------")
    print("Establecer el ranking de las alternativas, en orden descendente.")
    Alt = []
    PST.sort(reverse=True)
    print("reversa", PST)

    qqtemp = 0
    indxx = 0
    for i in range(a):
        au = 0
        for j in range(a):
            compar1 = round((float(PST[i])), 3)
            compar2 = round((float(PSS[au])), 3)
            # print(compar1,"=", compar2)
            if (compar1 == compar2):
                if indxx == 0:
                    qqtemp = j
                    Alt.append(qqtemp+1)
                else:
                    qqtemp = j
                    if (qqtemp+1) in Alt:
                        aalt = 0
                    else:
                        Alt.append(qqtemp+1)
                indxx = indxx+1
            au = au+1
            # print("Alt",Alt)
    print("   Producto sucesivo=   ", PST)
    print("   Ranking_alternativas=", Alt, "\n")

    # **********************************************************************PRIMERA ITERACIÓN
    print("PRIMERA ITERACIÓN -----------------------")
    print("w(inertia) = ", wwi)
    print("c1 = ", c1)
    print("c2 = ", c2)
    print("No. de iteraciones = ", T, "\n")
    print("r1 = ")
    print(r1, "\n")
    print("r2 = ")
    print(r2, "\n")
    print(
        "Función Objetivo: IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}")
    print("Rango de valores: (", rangoMin, ",", rangoMax, ") \n")

    # CURRENT VELOCITY (V)
    V = pd.DataFrame(columns=['C1', 'C2', 'C3', 'C4', 'C5'])
    for i in range(a):
        Vram1 = []
        for j in range(n):
            Vram = random.uniform(rangoMin, rangoMax)
            # print("Vram",Vram)
            Vram1.append(round((float(Vram)), 3))
        Ver1 = pd.DataFrame({'C1': [Vram1[0]], 'C2': [Vram1[1]], 'C3': [
                            Vram1[2]], 'C4': [Vram1[3]], 'C5': [Vram1[4]]})
        V = pd.concat([V, Ver1], ignore_index=True)
    print("V(1)=")
    print(V, "\n")

    # CURRENT POSITION (CP)
    # LA PRIMERA MEJOR POSICIÓN, SIEMPRE SERA LA PRIMERA POSICIÓN, NO SE TIENE ANTEDECENTES
    print("CP(1)=")
    print(CP, "\n")

    # FUNCIÓN OBJETIVO, CURRENT FITNESS (CF = Fx)
    # print("   Evaluar la función objetivo para obtener el mejor local y global.")
    # print("      Función Objetivo: IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}")
    # print("Establecer la solución ideal")
    # a)Solución ideal
    St = []
    for j in range(n):
        P1 = 0
        for i in range(a):
            P1 = float(P1)+float(A1.iat[i, j])
        P1 = round((float(P1)/float(a)), 3)
        St.append(float(P1))
    # print("St")
    # print(St)
    r1S = pd.Series(St)
    r1 = pd.concat([r1, r1S], ignore_index=True)
    # print("r1-Actualizado")
    # print(r1)

    # print("-------------------------------------------")
    # print("Determinar el índice de similitud")
    # a) normalizamos (a/S)
    # b) elevar lo normalizado al peso(w)
    # c) Producto sucesivo
    CFt = []
    SI1 = []
    PST = []
    for j in range(n):
        SI1 = []
        for i in range(a):
            dat1 = float(A1.iat[i, j])
            dat2 = float(S[j])
            wn2 = float(w[j])
            dat3 = round((dat1/dat2), 3)
            # print(dat1,dat2)
            # print(dat3,wn2)
            dat4 = abs(dat3)**abs(wn2)
            # print("        normalizado",round((dat3),3))
            # print("        Elevado",round((dat4),3))
            SI1.append(float(dat4))
        # print()
        # print("\n",SI1)
        Sqq1 = 1
        for z in range(a):
            # print(Sqq1)
            Sqq1 = (float(Sqq1)*float(SI1[z]))
        Sqq1 = round(Sqq1, 3)
        # print("producto",Sqq1)
        # print()
        CFt.append(float(Sqq1))
        # print("CFt")
        # print(CFt)
        # print("************************************************")
    r2S = pd.Series(CFt)
    r2 = pd.concat([r2, r2S], ignore_index=True)
    # print("CF(1)=")
    CF = pd.Series(CFt)
    # print("CF")
    # print(CF)

    CFPS = []
    for j in range(a):
        dat1 = 0
        for i in range(n):
            dat1 = dat1+float(CP.iat[j, i])
        dat1 = round((dat1/n), 3)
        CFPS.append(float(dat1))
    Fx = pd.Series(CFPS)
    # print("LBF=")
    # print(Fx,"\n")

    # LOCAL BEST POSITION OF EACH PARTICLE UP TO FIRST ITERATION IS JUST ITS CURRENT POSITION
    # SINCE THER IS NO PREVIUO ITERATION EXISTS
    # print("PBEST")
    # print(PBEST,"\n")
    # print("Fx(1)=")
    # print(Fx,"\n")
    pbestt = float(Fx.max())
    print("pbest(1)=", pbestt, "\n")

    # GLOBAL BEST FITNESS OF ITERATION #1
    GBF = []
    GBF.append(pbestt)
    print("GBF(1)=", GBF, "\n")

    # GLOBAL BEST POSITION OF ITERATION 1
    Fx_index = 0
    for j in range(a):
        val1 = round(float(GBF[0]), 3)
        val2 = round(float(Fx[j]), 3)
        # print(val1,"=",val2)
        if (val1 == val2):
            Fx_index = j
        # print("Fx_index",Fx_index)


    columna = []
    for i in range(n):
        Fx_P1 = float(A1.iat[Fx_index, i])
        columna.append(Fx_P1)
        # print(Fx_P1)
    # print(columna)
    GBP = pd.DataFrame(columna)
    print("gbest(1)=")
    print(GBP)
    Resultados.append(Fx_index+1)
    print("           Mejor alternativa= A", Fx_index+1, " para la iteración 1")
    print("      ---------------------------------------------------------------")


    ###############################################################################
    # ITERATION 2 a N
    t = 1
    par = 0
    longV1 = 0
    longseg = 5
    iii = 0

    while (t < T):

        print("\n ITERACIÓN #", t+1, "-----------------------", "\n")
        print("w(inertia) = ", wwi)
        print("c1 = ", c1)
        print("c2 = ", c2)
        print("No. de iteraciones = ", T, "\n")
        print("r1 (Actualizado) = ")
        print(r1.iloc[(len(r1)-n):len(r1)], "\n")
        print("r2 (Actualizado) = ")
        print(r2.iloc[(len(r2)-n):len(r2)], "\n")
        print(
            "Función Objetivo: IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}")
        print("Rango de valores: (", rangoMin, ",", rangoMax, ") \n")
        Fxce = []
        ii = 0
        longVel = a*t
        # print("tr12",tr12)
        for j in range(a):
            otroV = []
            otroCP = []
            tr12 = (len(r1)-n)
            CAA = (len(CP)-a)
            GBP12 = (len(GBP)-n)
            # print("                                                  CAA",CAA)
            # print("                                                  tr12",tr12)
            # print("                                                  GBP12",GBP12)
            for i in range(n):
                # print("                                             j,i ", j ,i)
                # 1-a) ACTUALIZANDO LA VELOCIDAD
                # Vtt1=0
                Vtt1 = float(V.iat[CAA, i])
                Vt11 = float((wwi*Vtt1))
                # print("Vt11",round((Vt11),3))

                PBESTtt = float((PBEST.iat[CAA, i]))
                rr1 = float(r1.iat[tr12])
                CPtt = float((CP.iat[CAA, i]))
                Vt12 = float((c1*rr1*(PBESTtt-CPtt)))
                # print("c1",c1)
                # print("rr1",rr1)
                # print("PBESTtt",PBESTtt)
                # print("CPtt",CPtt)
                # print("Vt12",round((Vt12),3))

                GBPtt = float(GBP.loc[GBP12])
                rr2 = float(r2.iat[tr12])
                # print("c2",c2)
                # print("rr2",rr2)
                # print("GBPtt",GBPtt)
                # print("CPtt",CPtt)
                Vt13 = float((c2*rr2*(GBPtt-CPtt)))
                # print("Vt13",round((Vt13),3))
                # print("--- \n")

                VFn = round((float(Vt11+Vt12+Vt13)), 3)
                # print("VFn",round((VFn),3))
                otroV.append(float(VFn))

                # 2-a) ACTUALIZANDO LA PRIMERA  POSICIÓN
                CPtt2 = float((CP.iat[CAA, i]))
                # print(CPtt2)
                # print(VFn)
                CPFn = round((float(VFn)+float(CPtt2)), 3)

                # print("CPFn",round((CPFn),3),"\n ")
                # 2-b) Verificar el rango de los valores
                if CPFn < rangoMin:  # <-5
                    CPFn = (rangoMin)+.2
                if CPFn > rangoMax:  # >5
                    CPFn = (rangoMax)-0.2
                # print(" --- Actualizado CPFn",CPFn)
                # print("+++++++++++++++++++++++++++ \n")
                otroCP.append(float(CPFn))
                tr12 = tr12+1
                GBP12 = GBP12+1

            V.loc[len(V.index)] = otroV
            CP.loc[len(CP.index)] = otroCP
            CAA = CAA+1

            # print("------ \n")
        # print(V)
        # print(CP)

        # 3-a) Evaluar la función objetivo para obtener el mejor local y global.
        # Función Objetivo: IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}
    # Es    tablecer la solución ideal
        St90 = []
        n_alt = 0
        for j in range(n):
            P1 = 0
            a_alt = a*t
            for i in range(a):
                CPtempt = CP.iat[a_alt, n_alt]
                # print("CPtempt",round((CPtempt),3))
                P1 = float(P1)+float(CPtempt)
                a_alt = a_alt+1
            P1 = round((float(P1)/float(a)), 3)
        # pr    int("P1",P1)
            St90.append(round((float(P1)), 3))
            # r1.loc[len(r1.index)]=P1
            n_alt = n_alt+1

        S0 = pd.Series(St90)
    # print("S0",round((S0),3))
        # print(r1)
        r1S = pd.Series(St90)
        r1 = pd.concat([r1, S0], ignore_index=True)

        # print("r1-Actualizado")
        # print(r1)

        # Determinar el índice de similitud")
        # a) normalizamos (a/S)
        # b) Elevar lo normalizado al peso (w)
        # c) Producto sucesivo
        xS0 = 0
        longA1 = 0
        CFt = []
        n_alt = 0
        for j in range(n):
            SI190 = []
            a_alt = a*t
            for i in range(a):
                dat1 = float(CP.iat[a_alt, n_alt])
                dat2 = float(S0[j])
                wn2 = float(w[j])
                # print(dat1, dat2)
                dat3 = dat1/dat2
                dat4 = abs(dat3)**abs(wn2)
                # print("        normalizado",round((dat3),3))
                # print("        Elevado",round((dat4),3))
                SI190.append(float(dat4))
                a_alt = a_alt+1
            n_alt = n_alt+1
            # print("------- \n")
            # print(SI190)
            Sqq1 = 1
            for z in range(a):
                Sqq1 = (float(Sqq1)*float(SI190[z]))
            Sqq1 = round((Sqq1), 3)
            # print("producto",Sqq1,"\n")
            CFt.append(float(Sqq1))
        r2S = pd.Series(CFt)
        r2 = pd.concat([r2, r2S], ignore_index=True)
        CFxe = pd.Series(CFt)
        CF = pd.concat([CF, CFxe], ignore_index=True)
        # print("r2")
        # print(r2)
        # print("CF=")
        # print(CF,"\n")

        # LOCAL BEST POSITION OF EACH PARTICLE UP TO FIRST ITERATION IS JUST ITS CURRENT POSITION
        # SINCE THER IS NO PREVIUO ITERATION EXISTS
        CFPS = []
        altFXx = (t*a)
        for j in range(a):
            dat1 = 0
            for i in range(n):
                # print("           ",float(CP.iat[altFXx,i]))
                dat1 = dat1+float(CP.iat[altFXx, i])
                # print("                    dat1",dat1)
            altFXx = altFXx+1
            dat1 = round((dat1/5), 3)  # entre 5 porque son 5 partículas
            # print("FINAL",dat1,"\n")
            CFPS.append(float(dat1))
        Fx12 = pd.Series(CFPS)
        Fx = pd.concat([Fx, Fx12], ignore_index=True)
        # print("LBF=")
        # print(Fx,"\n")

        zz1 = 0
        z1 = 0
        # print("CF")
        # print(CF)

        if t == 1:
            cont_act = n
            cont_ant = 0
        else:
            # print("longseg",longseg)
            cont_act = longseg-n
            cont_ant = cont_act-n

        for j in range(n):
            longsegP = len(CP)-a
            # print("                                  longsegP-INICIAL,z1",longsegP,z1)#18 (CP ACTUAL)

            # print(cont_ant,cont_act)
            actual = float(CF.iat[cont_act])
            anterior = float(CF.iat[cont_ant])
            # print("actual",actual, "anterior",anterior)
            LxCP = []

            # print(actual,">" ,anterior)
            if (actual > anterior) or (actual == anterior):  # CP(2)
                # print("entre al IF")
                for z in range(a):
                    # print("                                  longsegP-USADO,i",longsegP,z1)#9
                    x1 = CP.iat[longsegP, z1]
                    LxCP.append(round((x1), 3))
                    # print("         CP ACtual",round((x1),3))
                    longsegP = longsegP+1
            else:  # CP(1)
                # print("entre al ELSE")
                for z in range(a):
                    longsegPt = longsegP-a
                    # print("                                  longsegP-USADO",longsegP)#9
                    # print("                                     longsegPt-USADO,i",longsegPt,z1)#9
                    x1 = CP.iat[longsegPt, z1]
                    LxCP.append(round((x1), 3))
                    # print("CP anterior",round((x1),3))
                    longsegP = longsegP+1

            z1 = z1+1
            cont_ant = cont_ant+1
            cont_act = cont_act+1
            # print(LxCP)
            if j == 0:
                Cc1 = LxCP
                # print("Guarde eb cc1")
            if j == 1:
                Cc2 = LxCP
                # print("Guarde eb cc2")
            if j == 2:
                Cc3 = LxCP
                # print("Guarde eb cc3")
            if j == 3:
                Cc4 = LxCP
                # print("Guarde eb cc4")
            if j == 4:
                Cc5 = LxCP
                # print("Guarde eb cc5")
        new_CPLxCont = pd.DataFrame()
        new_CPLxCont['C1'] = Cc1
        new_CPLxCont['C2'] = Cc2
        new_CPLxCont['C3'] = Cc3
        new_CPLxCont['C4'] = Cc4
        new_CPLxCont['C5'] = Cc5
        # print("new_CPLxCont")
        # print(new_CPLxCont)
        PBEST = pd.concat([PBEST, new_CPLxCont], ignore_index=True)
        # print("PBEST")
        # print(PBEST)
        # print()

        # GLOBAL BEST FITNESS OF ITERATION
        # print("Fx")
        # print(Fx12)
        pbestt2 = float(max(Fx12))
        GBF.append(pbestt2)

        # GLOBAL BEST POSITION OF ITERATION
        Fx_index = 0
        temp_GBP = len(Fx)-a
        for j in range(a):
            val1 = round(float(GBF[t]), 3)
            val2 = round(float(Fx[temp_GBP]), 3)
            # print(val1,"=",val2)
            if (val1 == val2):
                Fx_index = j
            temp_GBP = temp_GBP+1
        Fx_index = Fx_index+(a*t)

        # print(CP)
        # print("++++Entre")
        columna = []
        for i in range(n):
            # print("Fx_index,i",Fx_index,i)
            Fx_P1 = float(CP.iat[Fx_index, i])
            columna.append(Fx_P1)
            # print(Fx_P1)
        # print(columna)
        GBP12 = pd.Series(columna)
        GBP = pd.concat([GBP, GBP12], ignore_index=True)
        # print("gbest(1)=")
        # print(GBP,"\n")
        Fx_index = Fx_index-(a*t)
        # print(Fx_index)
        Resultados.append(Fx_index+1)
        # print("           Mejor alternativa= A", Fx_index+1," para la iteración 1")
        # print("      ---------------------------------------------------------------")
        # print()

        # IMPRESIÓN DE RESULTADOS
        seg = a*t  # 5*1=5
        print("V(", t+1, ") =")
        print(V.iloc[seg:seg+a, :], "\n")
        print("CP(", t+1, ") =")
        print(CP.iloc[seg:seg+a, :], "\n")
        print("pbest(", t+1, ") =")
        print(PBEST.iloc[seg:seg+a], "\n")
        print("GBF =", GBF[t], "\n")
        print("gbest(", t+1, ") =")
        print(GBP.iloc[(len(GBP)-n):len(GBP)], "\n")
        # Mejor=(Fx_index2+1)-(a*t) si fuera CP
        Mejor = (Fx_index+1)
        print("           Mejor alternativa= A", Mejor, " para la iteración", t+1)
        print("      ---------------------------------------------------------------")
        # Resultados.append(Mejor)
        ii = ii+1
        iii = iii+1
        t = t+1

    print()
    print()
    print("**************************")
    print("Resultados Finales")
    print("**************************")
    # print("   Mejor posición=")
    # print(GBP.iloc[(len(GBP)-n):len(GBP)],"\n")
    # print("   Mejor óptimo=", GBF[t-1], "\n")


    print("   Ranking_DA= A", Alt[0])
    print("   Ranking_alternativas=", Alt)
    print()
    print("   Iteración", "  Mejor_alternativa")
    print("  ---------------------------------")
    dd = 0
    for i in range(T):
        print("       ", i+1, "        ", "A", Resultados[i])
    print("  ---------------------------------")
    DI = {
    "w(inertia)": wwi,
    "c1": c1,
    "c2": c2,
    "No. de iteraciones": T,
    "Función Objetivo": ['IS(a_{1}^i, a_{2}^i, ...a_{m}^i) = PI_{j=1}^m(a^i / S_{l})^w_{j}'],
    "Función objetivo PSO": ['Min f(x_{1}, x_{2}) = (x_{1}^{2} + (x_{2})^{2})'],
    "Rango_Min": rangoMin,
    "Rango_Max": rangoMax
    }
    dataI = pd.DataFrame(DI)
    dataGBF = pd.DataFrame(GBF)
    dataGBP = pd.DataFrame(GBP)
    dataPST = pd.DataFrame(PST)
    dataAlt = pd.DataFrame(Alt)
    dataw = pd.DataFrame(w)
    dataResult = pd.DataFrame(Resultados)
    alternativas = Resultados[-10:]
    hora_fin = datetime.datetime.now()

    with pd.ExcelWriter('Experimentos2/DAPSO.xlsx', engine='xlsxwriter') as writer:
        dataI.to_excel(writer, sheet_name='Iniciales')
        r1.to_excel(writer, sheet_name='r1')
        r2.to_excel(writer, sheet_name='r2')
        dataw.to_excel(writer, sheet_name='w')
        A1.to_excel(writer, sheet_name='Matriz')
        S.to_excel(writer, sheet_name='Solución ideal')
        dataPST.to_excel(writer, sheet_name='Producto sucesivo')
        dataAlt.to_excel(writer, sheet_name='Ranking_alternativas')
        PSS.to_excel(writer, sheet_name='Índice de similitud')
        V.to_excel(writer, sheet_name='Velocity')
        CP.to_excel(writer, sheet_name='Position')
        PBEST.to_excel(writer, sheet_name='PBEST')
        Fx.to_excel(writer, sheet_name='Fx')
        dataGBF.to_excel(writer, sheet_name='GBF')
        dataGBP.to_excel(writer, sheet_name='gbest')
        dataResult.to_excel(writer, sheet_name='Resultados')

        
        print('Datos guardados el archivo:DAPSO.xlsx')
        print()

        print('Algoritmo PSO-DAPSO')
        print('Cantidad de iteraciones: ',T)
        print('hora_inicio: ',hora_inicio.time())
        print('fecha_inicio: ',fecha_inicio)
        print('hora_finalizacion: ', hora_fin.time())
        print('tiempo_ejecucion: ', hora_fin - hora_inicio)
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


