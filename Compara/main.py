# Comparativo de Algoritmos
# Doctorado en Tecnología
# Universidad Autónoma de ciudad Juárez
# ACtualizado 10-Mar-2024

import os
import pandas as pd
import numpy as np
import pingouin as pg
import openpyxl
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats
import warnings
warnings.filterwarnings("ignore")
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
import matplotlib.font_manager as font_manager
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openpyxl.utils.dataframe import dataframe_to_rows

# Ruta a la carpeta que contiene los archivos Excel
carpeta_experimentos = "Experimentos"
n= 9 #Alternativas
d= 5 #Criterios 


########################################################
# Diccionario para almacenar los tiempos de ejecución combinados y el tiempo promedio
tiempos_combinados = {}
promedio_tiempo = {}

# Variable para indicar si algún nombre base comienza con 'ACO'
contiene = 0

# Iterar sobre todos los archivos en la carpeta
for archivo_excel in os.listdir(carpeta_experimentos):
    if archivo_excel.endswith(".xlsx"):
        nombre_archivo = os.path.splitext(archivo_excel)[0]  # Obtener el nombre del archivo sin la extensión
        ruta_completa = os.path.join(carpeta_experimentos, archivo_excel)
        
        try:
            df_tiempos = pd.read_excel(ruta_completa, usecols=["Tiempo de ejecución"]) # Leer la hoja "Tiempos"
            nombre_base = ''.join([i for i in nombre_archivo if not (i.isdigit() or i == '_')]) # Obtener el nombre base del archivo (sin el número al final ni guiones bajos)
            
            # Verificar si el nombre base comienza con 'ACO'
            if nombre_base.startswith('ACO'):
                contiene = 1
            
            df_tiempos["Tiempo de ejecución"] = pd.to_timedelta(df_tiempos["Tiempo de ejecución"])  # Convertir los tiempos de ejecución al formato deseado
            tiempo_promedio = df_tiempos["Tiempo de ejecución"].mean() # Calcular el tiempo promedio
            
            # Agregar los tiempos y el tiempo promedio al diccionario
            if nombre_base in tiempos_combinados:
                tiempos_combinados[nombre_base].append(df_tiempos)
            else:
                tiempos_combinados[nombre_base] = [df_tiempos]
            
            promedio_tiempo[nombre_base] = tiempo_promedio
            
        except KeyError as e:
            print(f"Error en el archivo {archivo_excel}: {e} no encontrado.")
        except Exception as e:
            print(f"Error en el archivo {archivo_excel}: {e}")

df_tiempos_combinados = pd.DataFrame() # Crear un DataFrame para almacenar los tiempos de ejecución combinados

# Combinar los tiempos de ejecución de cada algoritmo en un solo DataFrame
for nombre_base, dfs_tiempos in tiempos_combinados.items():
    df_tiempos_combinados[nombre_base] = pd.concat(dfs_tiempos, ignore_index=True)["Tiempo de ejecución"]

# Eliminar la parte de la fecha de las columnas
df_tiempos_combinados = df_tiempos_combinados.apply(lambda x: x.dt.total_seconds())

# Imprimir los tiempos de ejecución combinados
print("\n----------------------------------")
print("Tiempos de ejecución por algoritmo:")
print(df_tiempos_combinados)

# Imprimir el tiempo promedio de ejecución
print("\nTiempo promedio de ejecución por algoritmo:")
print(promedio_tiempo)

# Imprimir si algún nombre base contiene 'ACO'
print("\nContiene 'ACO':", contiene)





########################################################
# DataFrame para almacenar los datos de la hoja 'Resultados' de cada archivo con el nombre del archivo
resultados_dataframes = []

# Iterar sobre todos los archivos en la carpeta
for archivo_excel in os.listdir(carpeta_experimentos):
    if archivo_excel.endswith(".xlsx"):
        nombre_archivo = os.path.splitext(archivo_excel)[0]  # Obtener el nombre del archivo sin la extensión
        ruta_completa = os.path.join(carpeta_experimentos, archivo_excel)
        
        try:
            df_resultados = pd.read_excel(ruta_completa, sheet_name="Resultados_ejecución", header=None, usecols=[2]) # Leer la hoja 'Resultados'
            df_resultados = df_resultados.iloc[1:] # Ignorar el primer renglón (encabezado)
            df_resultados.columns = [nombre_archivo] # Agregar una columna con el nombre del archivo
            resultados_dataframes.append(df_resultados) # Almacenar el DataFrame en la lista de resultados
        
        except:
            df_resultados = pd.read_excel(ruta_completa, sheet_name="Resultados", header=None, usecols=[1]) # Leer la hoja 'Resultados'
            df_resultados = df_resultados.iloc[1:] # Ignorar el primer renglón (encabezado)
            df_resultados.columns = [nombre_archivo] # Agregar una columna con el nombre del archivo
            resultados_dataframes.append(df_resultados) # Almacenar el DataFrame en la lista de resultados
            

### -- Concatenar todos los DataFrames en uno solo
df_resultados_completos = pd.concat(resultados_dataframes, axis=1)
print("\ndf_resultados_completos\n",df_resultados_completos)

# Definir una función para eliminar 'A' al inicio de los valores y convertirlos a entero si es posible
def eliminar_a_inicio(valor):
    if isinstance(valor, str) and valor.startswith('A'):
        valor_sin_a = valor[1:]  # Eliminar 'A' al inicio
        if valor_sin_a.isdigit():  # Verificar si el valor sin 'A' es convertible a entero
            return int(valor_sin_a)  # Devolver el valor como entero
        else:
            return valor  # Si no es convertible a entero, devolver el valor original
    else:
        return valor  # Devolver el valor sin cambios si no comienza con 'A'

# Aplicar la función a cada elemento del DataFrame
df_resultados_completos = df_resultados_completos.applymap(eliminar_a_inicio)

print("Resultados completos, son 'A':")
print(df_resultados_completos) # Mostrar el DataFrame resultante




###############################################################
### -- Contar cuántas veces aparece cada número en cada columna
columnas = df_resultados_completos.columns # Obtener los nombres de las columnas de df_resultados_completos
conteo_numeros_list = [] # Construir una lista para almacenar los DataFrames de conteo
for col in columnas:
    conteo_col = df_resultados_completos[col].value_counts().reindex(range(1, 10), fill_value=0) # Realizar el conteo de valores y rellenar con ceros para los valores que no estén presentes
    conteo_numeros_list.append(conteo_col)  # Agregar el DataFrame de conteo a la lista

conteo_numeros = pd.concat(conteo_numeros_list, axis=1) # Concatenar todos los DataFrames de conteo en uno solo
conteo_numeros.columns = columnas # Asignar los nombres de las columnas a conteo_numeros
conteo_numeros_T = conteo_numeros.T # Transponer el DataFrame
#print("\n----------------------------------")
print("\nConteo de Alternativa por experimento")
print(conteo_numeros_T)

cant_ceros = conteo_numeros_T.eq(0).sum(axis=1) # Contar los ceros en cada fila
cant_sol = conteo_numeros_T.ne(0).sum(axis=1) # Contar los valores distintos de cero en cada fila
Solu1 = pd.DataFrame({ 'Cant_ceros': cant_ceros, 'Cant_Sol': cant_sol })
print("\nCantidad de Soluciones por experimento:")
print(Solu1)


### -- Estadísticas de las soluciones por algoritmo y experimento
nombres_similares = Solu1.index.str.split('_').str[0] # Obtener los nombres similares en el índice
estadisticas = Solu1['Cant_Sol'].groupby(nombres_similares).apply(lambda x: x.value_counts().sort_index()) 
estadisticas_pivot = estadisticas.unstack().fillna(0) # tener la alternativa como columnasde cada algoritmo
print("\nCantidad de soluciones por Algoritmo:")
print(estadisticas_pivot)

porcentajes_estadisticas = (estadisticas_pivot.div(estadisticas_pivot.sum(axis=1), axis=0) * 100).round(2)
print("\nPorcentaje de soluciones por Algoritmo:")
print(porcentajes_estadisticas)


### -- Porcentaje de soluciones por algoritmo, en tres grupos
grupos_de_tres = [porcentajes_estadisticas.columns[i:i+3] for i in range(0, len(porcentajes_estadisticas.columns), 3)]
   # Calcular la suma de cada grupo de tres columnas
Soluciones = pd.DataFrame()
for grupo in grupos_de_tres:
    grupo_str = [str(col) for col in grupo]  # Convertir elementos del grupo a cadenas de texto
    Soluciones['-'.join(grupo_str)] = porcentajes_estadisticas[grupo].sum(axis=1)
print("\nRango de soluciones por algoritmo:")
print(Soluciones)


### -- Frecuencia de alternativa por algoritmo 
Frecuencias = conteo_numeros_T.groupby(conteo_numeros_T.index.str.split('_').str[0]).agg('sum')
print("\nFrecuencias:")
print(Frecuencias)

Frec_porcentajes = (Frecuencias.div(Frecuencias.sum(axis=1), axis=0) * 100).round(2) # este contiene los porcentajes
print("\nPorcentajes de Frecuencia:")
print(Frec_porcentajes)




### --- Convergencia
conver_cambio_data = [] # Inicializar una lista para almacenar los datos de convergencia de los resultados
for columna in df_resultados_completos.columns:
    # Obtener el último valor de la columna y su índice
    ultimo_valor = df_resultados_completos[columna].iloc[-1]
    ultimo_index = df_resultados_completos.index[-1]

    # Buscar el penúltimo cambio de valor
    penultimo_valor = None
    penultimo_index = None
    for index, valor in reversed(list(df_resultados_completos[columna].iloc[:-1].items())):
        if valor != ultimo_valor:
            penultimo_valor = valor
            penultimo_index = index + 1  # Sumamos 1 al índice para empezar la búsqueda después de este índice
            break

    # Si no se encuentra ningún cambio, asignar el valor y el índice del último elemento como anteúltimo
    if penultimo_valor is None:
        penultimo_valor = ultimo_valor
        penultimo_index = ultimo_index

    # Buscar el anteúltimo cambio de valor
    anteultimo_valor = None
    anteultimo_index = None
    if penultimo_index is not None:
        for index, valor in reversed(list(df_resultados_completos[columna].iloc[:penultimo_index-1].items())):
            if valor != penultimo_valor:
                anteultimo_valor = valor
                anteultimo_index = index + 1  # Sumamos 1 al índice para empezar la búsqueda después de este índice
                break

    # Si no se encuentra ningún cambio antes del penúltimo cambio, asignar el valor y el índice del penúltimo elemento como anteúltimo
    if anteultimo_valor is None:
        anteultimo_valor = penultimo_valor
        anteultimo_index = 1 

    # Agregar los datos de convergencia de los resultados a la lista
    conver_cambio_data.append({
        'No_Iteraciones': ultimo_index,
        'Última_Alternativa': ultimo_valor,
        'IniciaU_Iteración': penultimo_index,
        'Penúltima_Alternativa': penultimo_valor,
        'Anteúltima_Iteración': anteultimo_index,
        'Anteúltima_Alternativa': anteultimo_valor
    })

Conver_cambio = pd.DataFrame(conver_cambio_data, index=df_resultados_completos.columns) # Crear un DataFrame a partir de los datos recopilados  
print("Convergencias")  
print(Conver_cambio) # Imprimir el DataFrame resultante


### --  Agrupar por nombres similares del índice y calcular los promedios
Prom_cambio = Conver_cambio.groupby(lambda x: x.split('_')[0]).agg({'IniciaU_Iteración': 'mean', 'Anteúltima_Iteración': 'mean'})
Prom_cambio.columns = ['cambio1', 'cambio2'] # Renombrar las columnas

# Convertir los promedios a números enteros
Prom_cambio['cambio1'] = Prom_cambio['cambio1'].astype(int)
Prom_cambio['cambio2'] = Prom_cambio['cambio2'].astype(int)
print(Prom_cambio) # Imprimir el DataFrame resultante
    




if contiene==0:
    
    #######################
    ### Alpha de Cronbach

    ### -- Lista para almacenar los DataFrames de la hoja 'CP' o 'Posiciones' de cada archivo
    df_cp_por_archivo = []
    resultados_alpha = []  # Inicializar la lista de resultados_alpha
    alpha_dict = {}  # Inicializar el diccionario alpha_dict
    archivo_aco_presente = False  # Bandera para verificar si se encontró algún archivo con 'ACO'
 
    for archivo_excel in os.listdir(carpeta_experimentos):  # Obtener las últimas posiciones 
        if archivo_excel.endswith(".xlsx"):
            if 'ACO' in archivo_excel or archivo_excel.startswith('ACO') or archivo_excel.endswith('ACO.xlsx'):
                archivo_aco_presente = True
                continue  # Si 'ACO' está presente al inicio, al final o en el nombre del archivo, pasa al siguiente archivo sin ejecutar el código restante
            nombre_archivo = os.path.splitext(archivo_excel)[0]  # Obtener el nombre del archivo sin la extensión
            ruta_completa = os.path.join(carpeta_experimentos, archivo_excel)
            
            try:
                # Intentar leer la hoja 'CP' o 'Posiciones'
                try:
                    df_cp = pd.read_excel(ruta_completa, sheet_name="CP", skiprows=-9, usecols=lambda x: x != 'A')
                except:
                    # Si no se encuentra la hoja 'CP', intentar leer la hoja 'Posiciones'
                    df_cp = pd.read_excel(ruta_completa, sheet_name="Posiciones", skiprows=-9, usecols=lambda x: x != 'A')

                # Tomar las últimas filas de los datos, de acuerdo a las alternativas que se tengan
                n = 10  # Establece el valor de 'n' adecuadamente
                df_cp = df_cp.tail(n)
                df_cp.reset_index(drop=True, inplace=True) # Reiniciar los índices para que inicien con el número 1
                df_cp = df_cp.apply(pd.to_numeric, errors='coerce')  # Convertir todas las columnas a tipo numérico float
                df_cp = df_cp.dropna(how='any')   # Eliminar filas que contengan valores no numéricos
                
                # Verificar si hay suficientes datos para calcular el alfa de Cronbach
                if len(df_cp) > 1:
                    alpha_cp = pg.cronbach_alpha(df_cp) # Calcular el Alpha de Cronbach para el DataFrame df_cp
                    resultados_alpha.append((nombre_archivo, alpha_cp[0], alpha_cp[1])) # Almacenar el resultado del Alpha de Cronbach y el intervalo de confianza en la lista
                    alpha_dict[nombre_archivo] = alpha_cp[0] # Almacenar el resultado del Alpha de Cronbach en el diccionario
                    
                else:
                    print(f"No hay suficientes datos para calcular el Alpha de Cronbach para {nombre_archivo}")
            
            except KeyError as e:
                print(f"Error en el archivo {archivo_excel}: {e} no encontrado.")
            except Exception as e:
                print(f"Error en el archivo {archivo_excel}: {e}")


    ### -- Verificar si se encontró algún archivo con 'ACO'
    if not archivo_aco_presente:
        df_resultados = pd.DataFrame(resultados_alpha, columns=['Nombre_Archivo', 'Alpha_Cronbach', 'Intervalo_Confianza'])

        # Agrupar por 'Nombre_Archivo' y calcular el promedio de 'Alpha_Cronbach' e 'Intervalo_Confianza'
        promedio_resultados = df_resultados.groupby('Nombre_Archivo').mean()
        print("Promedio de Alpha de Cronbach y Intervalo de Confianza por Nombre_Archivo:")
        print(promedio_resultados)


        ### -- Alpha de Cronbach de la Matriz original
        archivos = os.listdir(carpeta_experimentos) # Obtener la lista de archivos en la carpeta 'Experimentos'
        primer_archivo = archivos[0] # Seleccionar el primer archivo de la lista
        ruta_archivo = os.path.join(carpeta_experimentos, primer_archivo) # Construir la ruta completa al archivo

        try:
            # Leer la pestaña 'Matriz_decisión' del primer archivo, seleccionando solo las últimas 5 columnas
            df_originales = pd.read_excel(ruta_archivo, sheet_name="Matriz_decisión", usecols=lambda x: x in ['C1', 'C2', 'C3', 'C4', 'C5'])
            
            # Imprimir el DataFrame creado
            #print("DataFrame 'Originales':")
            #print(df_originales)

        except Exception as e:
            print(f"Error al leer el archivo {primer_archivo}: {e}")
        ac_Original = pg.cronbach_alpha(df_originales) # Calcular el coeficiente de alfa de Cronbach
        print("Alpha de Cronbach de la Matriz original:", ac_Original)
        print()


        ### -- Evaluación de qué resultados son mejores
        mejores_resultados = []
        resultados_bajos = []
        resultados_preocupantes = []

        for archivo, alpha in alpha_dict.items():
            if alpha > 0:
                mejores_resultados.append(archivo)
            elif alpha == 0:
                resultados_bajos.append(archivo)
            else:
                resultados_preocupantes.append(archivo)

        print("\nEvaluación de los resultados:")
        print("Los resultados son mejores para:", ', '.join(mejores_resultados), "ya que sus valores son positivos.")
        print("Para", ', '.join(resultados_bajos), "los resultados sugieren una consistencia interna baja.")
        print("Para", ', '.join(resultados_preocupantes), "los resultados son preocupantes porque indican una consistencia interna inversa o negativa.")
        print("-------------------------\n")
        
        # Asegurar que todas las listas tengan la misma longitud
        max_length = max(len(mejores_resultados), len(resultados_bajos), len(resultados_preocupantes))
        mejores_resultados += [np.nan] * (max_length - len(mejores_resultados))
        resultados_bajos += [np.nan] * (max_length - len(resultados_bajos))
        resultados_preocupantes += [np.nan] * (max_length - len(resultados_preocupantes))

        # Crear un DataFrame con los datos a guardar
        df_interpreta_ac = pd.DataFrame({
            'Mejores resultados': mejores_resultados,
            'Resultados bajos': resultados_bajos,
            'Resultados preocupantes': resultados_preocupantes
        })

        # Crear un DataFrame con los datos a guardar
        df_interpreta_ac = pd.DataFrame({
            'Mejores resultados': mejores_resultados,
            'Resultados bajos': resultados_bajos,
            'Resultados preocupantes': resultados_preocupantes
        })
print()
print()



###################
## estadísticas de los resultados de los experimentos
###################

### --- Calcula la moda de cada columna en el DataFrame
        # Moda: La moda es el valor que aparece con más frecuencia en un conjunto de datos. Es simplemente el número que se repite más veces.
moda_por_columna = df_resultados_completos.mode()
#print("Moda por columna:")
#print(moda_por_columna)

### --- Calcula la varianza de cada columna en el DataFrame
        # La varianza es una medida que indica qué tan dispersos están los valores de un conjunto respecto a su media.
        # Nos dice qué tan lejos están los valores de la media.
        # Si la varianza es alta, los valores están más dispersos. Si es baja, están más cerca de la media.
varianza_por_columna = df_resultados_completos.var()
#print("\nVarianza por columna:")
#print(varianza_por_columna)

### --- Calcular la media por columna
#Media: Calcula la media de cada columna para obtener un valor representativo de los datos.
media_por_columna = df_resultados_completos.mean()
#print("Media por columna:")
#print(media_por_columna)

### --- Calcular la mediana por columna
#Mediana: Calcula la mediana de cada columna para tener una medida de tendencia central robusta frente a valores atípicos.
mediana_por_columna = df_resultados_completos.median()
#print("\nMediana por columna:")
#print(mediana_por_columna)

### --- Calcular la desviación estándar por columna
#Desviación estándar: Calcula la desviación estándar de cada columna para medir la dispersión de los datos con respecto a la media.
desviacion_estandar_por_columna = df_resultados_completos.std()
#print("\nDesviación estándar por columna:")
#print(desviacion_estandar_por_columna)

### --- Calcula el coeficiente de variación por columna en el DataFrame
#Coeficiente de variación: Calcula el coeficiente de variación para comparar la dispersión relativa de diferentes conjuntos de datos.
coef_var_por_columna = df_resultados_completos.std() / df_resultados_completos.mean()
#print("Coeficiente de variación por columna:")
#print(coef_var_por_columna)


# Crear un DataFrame con las estadísticas solicitadas
df_estadisticas = pd.DataFrame({
    "Varianza": varianza_por_columna,
    "Moda": moda_por_columna.transpose().iloc[0],  # Transponer la moda y seleccionar la fila 0
    "Media": media_por_columna,
    "Mediana": mediana_por_columna,
    "Desviación Estándar": desviacion_estandar_por_columna,
    "Coeficiente de variación":coef_var_por_columna
})
# Imprimir el DataFrame de estadísticas
print("\nEstadísticas:")
#print(df_estadisticas)
df_estadisticas_redondeado = df_estadisticas.round(4)
print(df_estadisticas_redondeado)


### --- Calcular la correlación entre columnas
#Correlación: Calcula la correlación entre las diferentes columnas para ver si hay alguna relación lineal entre ellas.
correlacion_entre_columnas = df_resultados_completos.corr()
print("\nCorrelación entre columnas:")
#print(correlacion_entre_columnas)
df_correlacion_colum_redondeado=correlacion_entre_columnas.round(4)
print(df_correlacion_colum_redondeado)


### --- Calcular percentiles (25%, 50%, 75%) para cada columna en el DataFrame
# Percentiles: Calcula percentiles para entender la distribución de los datos y identificar valores atípicos.
# Esto calculará los percentiles del 25%, 50% y 75% para cada columna en el DataFrame df_resultados_completos y los imprimirá en la consola. 
# Los percentiles indican el valor por debajo del cual se encuentra un cierto porcentaje de los datos. 
# Por ejemplo, el percentil del 25% (o primer cuartil) indica el valor por debajo del cual se encuentra el 25% de los datos, 
# el percentil del 50% (o mediana) indica el valor por debajo del cual se encuentra el 50% de los datos, y así sucesivamente.
percentiles = df_resultados_completos.quantile([0.25, 0.50, 0.75])
print("Percentiles (25%, 50%, 75%) por columna:")
print(percentiles)


### --- Análisis de sensibilidad: 
#Realiza análisis de sensibilidad para evaluar cómo cambian los resultados cuando se modifican ciertos parámetros o condiciones.
# Este análisis puede ser útil para entender la estabilidad de los resultados y cómo varían en respuesta a cambios en los datos o condiciones del experimento. 
# Por ejemplo, un coeficiente de variación alto puede indicar una gran variabilidad en los datos, lo que puede afectar la fiabilidad de los resultados. 
# Por otro lado, un coeficiente de variación bajo sugiere una mayor consistencia en los datos. 
media2 = df_resultados_completos.mean() # Calcula la media de cada columna en el DataFrame
desviacion_estandar2 = df_resultados_completos.std() # Calcula la desviación estándar de cada columna en el DataFrame

#El coeficiente de variación de cada columna, calculado como la desviación estándar dividida por la media
coeficiente_de_variacion2 = desviacion_estandar2 / media2  # Calcula el coeficiente de variación de cada columna en el DataFrame
    
# Crear un DataFrame con las estadísticas solicitadas
df_estadisticas2 = pd.DataFrame({
    "Media": media2,
    "Desviación Estándar": desviacion_estandar2,
    "Coeficiente de variación": coeficiente_de_variacion2
})
df_sensitivity_analysis = df_estadisticas2.round(4)
print("\nAnálisis de sensibilidad:")
print(df_sensitivity_analysis)
    # Realiza el análisis de sensibilidad
        #Para interpretar los resultados del análisis de sensibilidad:
        #Media: Examina las medias para entender los valores típicos en cada columna. Si hay diferencias significativas entre las medias de diferentes columnas, indica que los valores promedio varían considerablemente entre esas columnas.
        #Desviación estándar: Evalúa las desviaciones estándar para entender la dispersión de los datos alrededor de la media en cada columna. Una desviación estándar alta sugiere una mayor dispersión de los datos, lo que puede indicar una mayor variabilidad o incertidumbre en esos datos.
        #Coeficiente de variación: Analiza el coeficiente de variación para entender la variabilidad relativa en relación con la media en cada columna. Un coeficiente de variación alto sugiere una alta variabilidad relativa en comparación con la media, lo que puede indicar una mayor heterogeneidad en los datos.



### --- Análisis de robustez: Evalúa la robustez de tus conclusiones mediante análisis de sensibilidad o pruebas de replicación en diferentes conjuntos de datos.
# El análisis de sensibilidad es una técnica utilizada para evaluar cómo cambian los resultados de un modelo o experimento cuando se modifican ciertos parámetros o condiciones. 
    # Esto ayuda a comprender la robustez y la fiabilidad de los resultados ante posibles variaciones en los datos o supuestos.
    # El análisis de robustez es una evaluación de la estabilidad y fiabilidad de las conclusiones o resultados de un experimento o modelo. En el contexto del análisis de sensibilidad que proporcionaste, el objetivo es determinar qué tan robustas son las conclusiones frente a variaciones en los datos o supuestos.
    # Algunos pasos que podrías seguir para realizar este análisis:
# 1) Coeficiente de variación: 
        #Calcula el coeficiente de variación para cada columna, que es la relación entre la desviación estándar y la media. Un coeficiente de variación bajo indica una mayor estabilidad en los resultados.
        # Este coeficiente mide la variabilidad relativa en relación con la media. Un coeficiente de variación bajo indica una mayor estabilidad en los resultados, ya que implica que la variabilidad es baja en comparación con la media. Por lo tanto, si el CV es bajo en todas las columnas, indica que los resultados son más robustos y consistentes.
coef_var2 = df_resultados_completos.std() / df_resultados_completos.mean()
#print("Coeficiente de variación", coef_var2)
# 2) Rango intercuartil: Calcula el rango intercuartil (IQR) para cada columna, que es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). Un IQR pequeño indica una menor variabilidad en los datos.
        #Rango intercuartil (IQR): El rango intercuartil es la diferencia entre el tercer cuartil (Q3) y el primer cuartil (Q1). Un IQR pequeño indica una menor variabilidad en los datos. Si el IQR es pequeño en todas las columnas, sugiere que los datos tienen una menor dispersión, lo que contribuye a la robustez de los resultados.
iqr2 = df_resultados_completos.quantile(0.75) - df_resultados_completos.quantile(0.25)
#print("Rngo de intercuartil",iqr2)
# 3) Correlaciones: Calcula las correlaciones entre las columnas para identificar posibles relaciones entre los resultados de diferentes experimentos.
    #Calcular las correlaciones entre las columnas te ayuda a identificar posibles relaciones entre los resultados de diferentes experimentos. Si hay correlaciones fuertes entre algunas columnas, podría indicar que ciertos factores están influenciando consistentemente los resultados. Esto puede ser útil para entender la estabilidad de las relaciones observadas en el experimento.    
correlaciones2 = df_resultados_completos.corr()
#print("correlaciones",correlaciones2)

df_estadisticas3 = pd.DataFrame({
    "Coeficiente de variación": coef_var2,
    "Rango intercuartil": iqr2,
    "Correlación entre columnas": correlaciones2.transpose().iloc[0],
})
df_Análisis_robustez = df_estadisticas3.round(4)
print("\nAnálisis de robustez:")
print(df_Análisis_robustez)
# Para interpretar los resultados de estos análisis en conjunto:
    #Si el coeficiente de variación es bajo en todas las columnas y el rango intercuartil es pequeño, indica que los resultados son consistentes y poco sensibles a las variaciones en los datos.
    #Si las correlaciones entre las columnas son consistentes y fuertes, sugiere que las relaciones observadas son robustas y no dependen en gran medida de las condiciones específicas del experimento.



### --- Calcula la similitud entre observaciones para identificar patrones de agrupamiento o para encontrar observaciones similares en un conjunto de datos.
# 1) Una opción común es calcular la matriz de correlación entre las columnas, que te dará una medida de la similitud lineal entre ellas.
# Esta matriz mostrará la correlación entre cada par de columnas en tu DataFrame. Valores cercanos a 1 indican una correlación positiva fuerte, valores cercanos a -1 indican una correlación negativa fuerte, y valores cercanos a 0 indican poca correlación.
# Calcular la matriz de correlación
matriz_correlacion = df_resultados_completos.corr()
print("Matriz de correlación:")
print(matriz_correlacion)
    #La matriz de correlación es una tabla que muestra la correlación entre cada par de columnas en tu DataFrame. Los valores de la matriz están en el rango de -1 a 1:
        #Valores cercanos a 1 indican una correlación positiva fuerte, lo que significa que las dos columnas tienden a aumentar o disminuir juntas.
        #Valores cercanos a -1 indican una correlación negativa fuerte, lo que significa que las dos columnas tienden a moverse en direcciones opuestas.
        #Valores cercanos a 0 indican poca correlación, lo que significa que no hay una relación lineal clara entre las dos columnas.
        #Al interpretar la matriz de correlación, busca patrones o grupos de columnas que tengan valores altos de correlación entre sí. Esto podría indicar que estas columnas están relacionadas de alguna manera o que muestran patrones similares en los datos. Por otro lado, si las correlaciones son cercanas a cero, indica que las columnas son independientes entre sí.

# 2) Otra opción es utilizar técnicas de análisis multivariado, como el análisis de componentes principales (PCA) o técnicas de clustering, para identificar grupos de columnas que sean similares entre sí en términos de sus valores.
    # Dependiendo de lo que necesites específicamente en tu análisis de similitud, podrías explorar otras técnicas más avanzadas o adaptar estas sugerencias según tus necesidades particulares.
    #  primero necesitaremos asegurarnos de que los datos estén preparados adecuadamente. Dado que PCA es una técnica que se utiliza para reducir la dimensionalidad de los datos y encontrar patrones en ellos, es importante que los datos estén estandarizados (es decir, centrados en cero y escalados).
# Estandarizar los datos
    # Antes de aplicar PCA, es importante estandarizar los datos para que tengan una media de cero y una desviación estándar de uno. Esto asegura que todas las variables tengan la misma escala y permite que PCA identifique correctamente la estructura de variación en los datos.
scaler = StandardScaler()
df_resultados_estandarizados = scaler.fit_transform(df_resultados_completos)

# Crear una instancia de PCA y ajustarla a los datos estandarizados
    #Después de estandarizar los datos, PCA encuentra una serie de componentes principales que son combinaciones lineales de las variables originales. Cada componente principal captura una parte de la variabilidad total en los datos. Los componentes principales están ordenados de mayor a menor importancia en términos de la cantidad de varianza que explican
pca = PCA()
pca.fit(df_resultados_estandarizados)

# Obtener los componentes principales y las proporciones de varianza explicada
    #Para cada componente principal, PCA también calcula la proporción de varianza total en los datos que se explica por ese componente. Estas proporciones de varianza explicada te dicen cuánta información captura cada componente principal en comparación con la varianza total en los datos.
componentes_principales = pca.components_
varianza_explicada = pca.explained_variance_ratio_

# Imprimir los componentes principales y las proporciones de varianza explicada
    # Al interpretar los resultados de PCA:
        # Componentes principales: Examina los componentes principales para identificar patrones o estructuras en los datos. Los componentes principales que tienen coeficientes grandes para ciertas variables indican que esas variables contribuyen más a ese componente en particular.
        # Proporciones de varianza explicada: Observa las proporciones de varianza explicada para entender cuánta información captura cada componente principal. Los componentes principales con proporciones de varianza explicada más altas son más importantes para describir la variabilidad en los datos.
print("Componentes principales:")
print(componentes_principales)
print("\nProporciones de varianza explicada:")
print(varianza_explicada)


### --- Análisis de desempeño: Evalúa el desempeño de ciertos procesos o sistemas en función de los datos recopilados.
#Esto calculará estadísticas clave como la media, la mediana, el mínimo, el máximo y los cuartiles para cada columna del DataFrame. Además, generará un diagrama de caja para visualizar la distribución de los resultados de cada algoritmo.
# Este análisis te dará una idea general del desempeño de los algoritmos, incluyendo la dispersión de los resultados y cualquier posible tendencia central.
# Calcular estadísticas clave
estadisticas_desempeño = df_resultados_completos.describe()
# Imprimir las estadísticas
print("\nEstadísticas de desempeño:")
print(estadisticas_desempeño)
    # Interpretación de las estadísticas:
        # Estadísticas clave: El código utiliza el método describe() para calcular estadísticas clave para cada columna del DataFrame. Estas estadísticas incluyen la media, la mediana, el mínimo, el máximo y los cuartiles (25%, 50%, 75%) de los datos en cada columna.
        # Media: Es el promedio de los valores en cada columna. Indica la tendencia central de los datos.
        # Mediana: Es el valor que se encuentra en el medio de un conjunto de datos ordenados. Es menos sensible a valores extremos que la media.
        # Mínimo y máximo: Representan los valores más pequeños y más grandes en cada columna, respectivamente.
        # Cuartiles: Son los valores que dividen los datos ordenados en cuatro partes iguales. El cuartil 50% es la mediana, el cuartil 25% es el valor por debajo del cual cae el 25% de los datos y el cuartil 75% es el valor por debajo del cual cae el 75% de los datos.




###########################################################
###########################################################
        
# Guardar los DataFrames en un archivo de Excel
        
##########################################################
##########################################################
# Verificar los tipos de datos de los DataFrames
df_correlacion_colum_redondeado = df_correlacion_colum_redondeado.apply(pd.to_numeric, errors='coerce')
matriz_correlacion = matriz_correlacion.apply(pd.to_numeric, errors='coerce')

fecha_hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Obtener la fecha y hora actual
with pd.ExcelWriter('Comparaciones.xlsx', engine='xlsxwriter') as writer:

# Guardar la correlación entre columnas y la matriz de correlación
    #df_correlacion_colum_redondeado.to_excel(writer, sheet_name='Correlacion_columnas', index=True, header=True)
    #matriz_correlacion.to_excel(writer, sheet_name='Matriz_correlacion', index=True, header=True)
    
    df_resultados_completos.to_excel(writer, sheet_name='Resultados')
    df_tiempos_combinados.to_excel(writer, sheet_name='Tiempos')
    df_promedio_tiempo = pd.DataFrame(promedio_tiempo.items(), columns=["Algoritmo", "Tiempo promedio"])
    # Convertir los valores del tiempo promedio a segundos como flotantes
    valores_tiempo_promedio = df_promedio_tiempo["Tiempo promedio"]
    valores_tiempo_promedio_numericos = valores_tiempo_promedio.apply(lambda x: pd.Timedelta(x).total_seconds()).astype(float)
    df_promedio_tiempo["Tiempo promedio"] = valores_tiempo_promedio_numericos
    df_promedio_tiempo.to_excel(writer, sheet_name='Tiempo_promedio')

    conteo_numeros_T.to_excel(writer, sheet_name='Conteo_Alternativas')
    Solu1.to_excel(writer, sheet_name='Cantidad_Soluciones')
    estadisticas_pivot.to_excel(writer, sheet_name='Soluciones_Por_Algoritmo')
    porcentajes_estadisticas.to_excel(writer, sheet_name='Porcentaje_Sol_Algoritmo')
    Soluciones.to_excel(writer, sheet_name='Rango_Soluciones')

    Frecuencias.to_excel(writer, sheet_name='Frecuencias_Alternativas')
    Frec_porcentajes.to_excel(writer, sheet_name='Porcentaje_Frec_Alterna')

    Conver_cambio.to_excel(writer, sheet_name='Convergencia')
    Prom_cambio.to_excel(writer, sheet_name='Promedio_convergencia')

    if contiene==0:


        # Extraer el valor del coeficiente de alfa de Cronbach y el intervalo de confianza de la tupla
        coeficiente_alfa = ac_Original[0]
        intervalo_confianza = ac_Original[1]
        
        # Crear un DataFrame con el coeficiente de alfa de Cronbach y el intervalo de confianza
        df_alfa_original = pd.DataFrame({ 'Alpha de Cronbach de la Matriz original': [coeficiente_alfa], 'Intervalo de Confianza': [intervalo_confianza] })
        
        # Guardar los DataFrames en el archivo de Excel
        df_alfa_original.to_excel(writer, sheet_name='Alpha_Cronbach_Original')
        df_resultados.to_excel(writer, sheet_name='Alpha_Cronbach_Resultados')
        df_interpreta_ac.to_excel(writer, sheet_name='Interpretacion_Alpha_Cronbach')
    
    #Datos de estadísticas de los resultados de cada experimento
    estadisticas_desempeño.to_excel(writer, sheet_name='estadisticas_desempeño')
    df_estadisticas_redondeado.to_excel(writer, sheet_name='Estadíticas')
    percentiles.to_excel(writer, sheet_name='percentiles')
    df_sensitivity_analysis.to_excel(writer, sheet_name='Análisis_sensibilidad')
    df_Análisis_robustez.to_excel(writer, sheet_name='Análisis_robustez')

    if isinstance(componentes_principales, np.ndarray):
        componentes_principales = pd.DataFrame(componentes_principales)
    componentes_principales.to_excel(writer, sheet_name='Componentes_principales')
  
    workbook = writer.book
    hoja_prop_var_explicada = workbook.add_worksheet('Proporciones_varianza_explicada')
    for i, valor in enumerate(varianza_explicada, start=1):
        hoja_prop_var_explicada.write(i - 1, 0, f"Componente {i}")
        hoja_prop_var_explicada.write(i - 1, 1, valor)
    
    df_fecha_hora = pd.DataFrame({'Fecha y hora de ejecución': [fecha_hora_actual]})
    df_fecha_hora.to_excel(writer, sheet_name='Fecha_Hora_Ejecucion')



    ############################
    # GRÁFICAS
    ############################

    ##############
    ### -- Gráfica de la pestaña 'Tiempo_promedio'
    sheet_tiempo_promedio = writer.sheets['Tiempo_promedio']    
    df_promedio_tiempo_sorted = df_promedio_tiempo.sort_values(by=df_promedio_tiempo.columns[1]) # Ordenar el DataFrame por los valores de tiempo promedio

    # Obtener los nombres de los algoritmos y los valores de tiempo promedio ordenados
    nombres_algoritmos = df_promedio_tiempo_sorted.iloc[:, 0]
    valores_tiempo_promedio_numericos = df_promedio_tiempo_sorted.iloc[:, 1].astype(float)

    colores_series = ['orchid', 'green', 'Gray', 'skyblue','orange', 'purple', 'magenta', 'Brown','Pink', 'peru','indigo', 'darkorange', 'chocolate'] 
    plt.figure(figsize=(7, 5))  # Tamaño reducido de la figura
    bars = plt.bar(nombres_algoritmos, valores_tiempo_promedio_numericos, color=colores_series)  # Crear la gráfica de barras
    plt.grid(axis='y', linestyle='-', linewidth=0.5) # Agregar líneas de cuadrícula horizontal principal primaria

    for bar in bars: # Agregar etiquetas con los valores de las barras
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom', fontweight='bold')

    plt.title('Tiempo promedio por algoritmo', fontsize=20, fontweight='bold', pad=20)  
    plt.xlabel('Ejecuciones del programa',fontweight='bold')
    plt.ylabel('Tiempo promedio (segundos)',fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('grafica_tiempo_promedio.jpg', format='jpg', dpi=300)  # Guardar la gráfica en un archivo
    plt.close()
    sheet_tiempo_promedio.insert_image('F3', 'grafica_tiempo_promedio.jpg')  # Insertar la imagen en la hoja de Excel




    ##############
    # Generar la gráfica 'Porcentajes Soluciones'
    plt.figure(figsize=(12, 10))  # Aumentar el tamaño de la figura
    ax = plt.gca()  # Obtener el objeto del eje actual
    colores_series = ['darkorange', 'slategray', 'green', '#f7b6d2', '#c5b0d5', '#e7ba52', '#8c6d31', 'skyblue', 'brown']
    bars = porcentajes_estadisticas.plot(kind='bar', stacked=True, ax=ax, color=colores_series[:len(porcentajes_estadisticas.columns)],width=0.6)

    plt.title('Porcentaje de soluciones por Algoritmo', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Algoritmo', fontweight='bold',fontsize=14)
    plt.ylabel('Porcentaje de Soluciones', fontweight='bold',fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)  # Ajustar el espaciado en la parte inferior para hacer espacio para la tabla de datos

    # Agregar una tabla de datos
    table_data = [['{:.2f}'.format(value) for value in row] for row in porcentajes_estadisticas.values]
    table = plt.table(cellText=table_data,
                        colLabels=porcentajes_estadisticas.columns,
                        rowLabels=porcentajes_estadisticas.index,  # Mantenemos las etiquetas de fila sin formato
                        loc='bottom', bbox=[0.1, -0.6, 0.8, 0.4],  # Ajusta la posición y el tamaño de la tabla
                        cellLoc='right',
                        rowLoc='left')

    # Colorear los encabezados de la tabla con los colores definidos
    for col, color in zip(porcentajes_estadisticas.columns, colores_series):
        cell = table[(0, porcentajes_estadisticas.columns.get_loc(col))]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold', color='black')  # Cambiar el texto a negrita y blanco

    # Establecer el estilo de fuente para las etiquetas de fila
    for i in range(len(porcentajes_estadisticas.index)):
        cell = table[(i+1 ,-1)]  # Las etiquetas de fila comienzan desde la fila 1
        cell.set_text_props(weight='bold')  # Establecer el texto a negrita

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.5, 1.5)  # Ajusta este valor según sea necesario para mostrar completamente la tabla

    # Cuadro de leyenda
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='No_soluciones')
    font_prop = font_manager.FontProperties(weight='bold')  
    legend.get_title().set_fontproperties(font_prop)  
    plt.tight_layout()  # Ajustar el diseño de la gráfica para que todos los elementos se ajusten adecuadamente

    # Insertar la gráfica en el libro de  y crear en carpeta
    plt.savefig('grafica_porcentaje_Soluciones.jpg', format='jpg', dpi=300, bbox_inches='tight')  # Guardar la gráfica en un archivo JPG
    sheet_porcentaje_frecuencia = writer.sheets['Porcentaje_Sol_Algoritmo'] # Insertar la imagen en la hoja de Excel
    sheet_porcentaje_frecuencia.insert_image('L3', 'porcentaje_Soluciones.jpg', {'x_scale': 0.7, 'y_scale': 0.7})
    plt.close()  # Cerrar la figura para liberar memoria



    ##############
    ### -- Gráfica 'Rangos Soluciones'
    fig, ax = plt.subplots(figsize=(12, 10)) 
    colores_series = ['darkorange', 'teal', 'green'] 
    columnas_grafica = Soluciones.columns
    # Hacer la gráfica de barras apiladas
    bars = Soluciones[columnas_grafica].plot(kind='barh', ax=ax, color=colores_series, stacked=True, width=0.6)

    ax.grid(axis='x', linestyle='-', linewidth=0.5) # Cambiamos a grid vertical

    # Establecer título y etiquetas
    ax.set_title('Rango de soluciones por algoritmo', fontsize=20, fontweight='bold', pad=20, ha='center')
    ax.set_xlabel('Porcentaje de Soluciones', fontweight='bold')
    ax.set_ylabel('Algoritmo', fontweight='bold')
    plt.yticks(rotation=45, ha='right')    # Ajustar posición del eje y
    plt.tight_layout()  # Ajustar el diseño de la gráfica para que todos los elementos se ajusten adecuadamente
    plt.subplots_adjust(left=0.30) # Ajustar el espaciado en la parte izquierda para hacer espacio para la tabla de datos

    # Insertar una tabla de datos
    tabla = ax.table(cellText=Soluciones.round(0).values,
                    colLabels=columnas_grafica,
                    rowLabels=Soluciones.index,
                    loc='left', bbox=[0.3, -0.5, 0.5, 0.4])  # Ajusta la posición y el tamaño de la tabla

    # Colorear los encabezados de la tabla con los colores definidos
    for col, color in zip(Soluciones.columns, colores_series): 
        cell = tabla[(0, Soluciones.columns.get_loc(col))]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold', color='black')  # Cambiar el texto a negrita y blanco

    # Establecer el estilo de fuente para las etiquetas de fila
    for i in range(len(Soluciones.index)):
        cell = tabla[(i + 1, -1)]  # Las etiquetas de fila comienzan desde la fila 1
        cell.set_text_props(fontweight='bold')  # Establecer el texto a negrita

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(13)
    tabla.scale(1.2, 1.2)  # Escalar la tabla

    # Añadir título de las series como leyendas
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Rango_Soluciones')
    font_prop = font_manager.FontProperties(weight='bold')  
    legend.get_title().set_fontproperties(font_prop)  

    # Insertar la gráfica en el libro de  y crear en carpeta
    sheet = writer.sheets['Rango_Soluciones']
    fig.savefig('grafica_Rango_soluciones.jpg', format='jpg', dpi=300, bbox_inches='tight')  # Guardar la gráfica en un archivo JPG
    sheet.insert_image('G3', 'grafica_Rango_soluciones.jpg', {'x_scale': 0.7, 'y_scale': 0.7})  # Ajusta la posición vertical de la imagen
    plt.close() # Cerrar la figura para liberar memoria




    ##############
    ### -- Gráfica 'Promedio de Convergencia'
    fig, ax = plt.subplots()
    Prom_cambio.plot(kind='bar', ax=ax,width=0.8) # Graficar los promedios de convergencia
    ax.set_title('Promedio de Convergencia', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Algoritmos', fontweight='bold')
    ax.set_ylabel('Iteraciones', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='-', linewidth=0.5) # Agregar líneas de cuadrícula horizontal principal primaria

    # Añadir título de las series como leyendas
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Convergencia')

    # Establecer el título de la leyenda en negrita
    legend.set_title('Convergencia')
    legend.get_title().set_fontweight('bold')

    # Agregar los valores dentro de las barras
    for bar in ax.patches:
        bar_value = bar.get_height()
        bar_x = bar.get_x() + bar.get_width() / 2
        bar_y = bar.get_height()
        ax.text(bar_x, bar_y, f'{int(bar_value)}', ha='center', va='bottom', fontsize=8, fontweight='bold')  

    # Insertar la gráfica en el libro de  y crear en carpeta
    plt.savefig('grafica_porcentaje_convergencia.jpg', format='jpg', dpi=300, bbox_inches='tight') 
    sheet = writer.sheets['Promedio_convergencia'] # Insertar la imagen en la hoja de Excel
    sheet.insert_image('E3', 'grafica_porcentaje_convergencia.jpg')  # Ajusta la posición vertical de la imagen
    plt.close() # Cerrar la figura para liberar memoria


 


    ##############
    ### -- Gráfica 'Frecuencias'

    fig, ax = plt.subplots(figsize=(12, 10))
    colores_series = ['darkorange', 'slategray', 'green', '#f7b6d2', '#c5b0d5', '#e7ba52', '#8c6d31', 'skyblue', 'brown']
    bars = Frec_porcentajes.plot(kind='bar', stacked=True, ax=ax, color=colores_series[:len(Frec_porcentajes.columns)],width=0.6) 

    plt.title('Porcentaje de Frecuencia de Alternativas por Algoritmo', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Algoritmo', fontweight='bold')
    plt.ylabel('Porcentaje de Frecuencia', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30) # Ajustar el espaciado en la parte inferior para hacer espacio para la tabla de datos

    # Cuadro de leyenda
    handles, labels = ax.get_legend_handles_labels() # Obtener los manejadores y etiquetas de la leyenda antes de cerrar la figura
    legend=plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Alternativas') # Colocar la leyenda a la derecha y centrada
    font_prop = font_manager.FontProperties(weight='bold')  
    legend.get_title().set_fontproperties(font_prop)  
    plt.tight_layout()  # Ajustar el diseño de la gráfica para que todos los elementos se ajusten adecuadamente

    # Agregar una tabla de datos
    table_data = [['{:.2f}'.format(value) for value in row] for row in Frec_porcentajes.values]
    table = plt.table(cellText=table_data,
                        colLabels=Frec_porcentajes.columns,
                        rowLabels=Frec_porcentajes.index,
                        loc='bottom', bbox=[0.1, -0.55, 0.8, 0.4],  # Ajusta la posición y el tamaño de la tabla
                        cellLoc='right',
                        rowLoc='left')
    
    for col, color in zip(Frec_porcentajes.columns, colores_series): # Colorear los encabezados de la tabla con los colores definidos
        cell = table[(0, Frec_porcentajes.columns.get_loc(col))]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold', color='black')  # Cambiar el texto a negrita y blanco

    # Establecer el estilo de fuente para las etiquetas de fila
    for i in range(len(Frec_porcentajes.index)):
        cell = table[(i + 1, -1)]  # Las etiquetas de fila comienzan desde la fila 1
        cell.set_text_props(fontweight='bold')  # Establecer el texto a negrita

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.6, 1.6)  # Ajusta este valor según sea necesario para mostrar completamente la tabla

    # Insertar la gráfica en el libro de  y crear en carpeta
    plt.savefig('grafica_porcentaje_frecuencia.jpg', format='jpg', dpi=300, bbox_inches='tight')  # Guardar la gráfica en un archivo JPG
    sheet_porcentaje_frecuencia = writer.sheets['Porcentaje_Frec_Alterna'] # Insertar la imagen en la hoja de Excel
    sheet_porcentaje_frecuencia.insert_image('L3', 'grafica_porcentaje_frecuencia.jpg', {'x_scale': 0.7, 'y_scale': 0.7})
    plt.close() # Cerrar la figura para liberar memoria


print("\nArchivo Guardado\n")
