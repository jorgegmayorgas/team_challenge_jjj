import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import ast
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind,mannwhitneyu,ttest_rel,ttest_1samp
from scipy.stats import chi2_contingency,f_oneway

def get_features_num_classification (dataframe:pd.DataFrame,target_col:str,pvalue:float=0.5):
    """
    Función que devuelve un Dataframe  de Pandas con:

        * El tipo de datos por columna
    
        * El porcentaje de nulos
        
        * La cantidad de valores únicos
        
        * La cardinalidad de la columna en porcentaje

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas a describir.
    

    Retorna:
    pandas.DataFrame: Dataframe
    """
    return None

def plot_features_num_classification (dataframe:pd.DataFrame,target_col:str="",columns:list=[],pvalue:float=0.5):
    """
    Función que devuelve un Dataframe  de Pandas con:

        * El tipo de datos por columna
    
        * El porcentaje de nulos
        
        * La cantidad de valores únicos
        
        * La cardinalidad de la columna en porcentaje

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas a describir.
    

    Retorna:
    pandas.DataFrame: Dataframe
    """
    return None

#version inicial toolbox_ML.py
def describe_df(df:pd.DataFrame):
    """
    Función que devuelve un Dataframe  de Pandas con:

        * El tipo de datos por columna
    
        * El porcentaje de nulos
        
        * La cantidad de valores únicos
        
        * La cardinalidad de la columna en porcentaje

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas a describir.
    

    Retorna:
    pandas.DataFrame: Dataframe
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"El parámetro debe ser un Dataframe de Pandas")
    tipo_de_dato = df.dtypes
    valores_nulos = (df.isna().mean() * 100).round(2)
    valores_unicos = df.nunique()
    cardinalidad = ((df.nunique()/df.shape[0])*100).round(2)

    descripcion = pd.DataFrame({
        'Tipo de dato': tipo_de_dato,
        'Valores nulos (%)': valores_nulos,
        'Valores unicos': valores_unicos,
        'Cardinalidad (%)': cardinalidad
    })

    return descripcion.T
    

def get_features_num_regression(df:pd.DataFrame, target_col:str, umbral_corr:float, pvalue=None):
    """
    Función que devuelve una lista con las columnas numéricas 
    del dataframe cuya correlación con la columna designada 
    por "target_col" sea superior en valor absoluto al valor dado 
    por "umbral_corr". 
    
    Además si la variable "pvalue" es distinta de None, sólo devolvera 
    las columnas numéricas cuya correlación supere el valor indicado y 
    además supere el test de hipótesis con significación 
    mayor o igual a 1-pvalue

    Argumentos:
    
    `df` (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    `target_col` (list): Nombre de la columna target de un modelo de regresión
    
    `umbral_corr` (float): Umbral de correlación, entre 0 y 1.
    
    `pvalue` (float): Descripción de param1.

    Retorna:
    
    list: Lista de Python
    """
    # Verificamos el tipo en la variable df
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"El primer parámetro debe ser un Dataframe de Pandas")
    # Verificamos el tipo en la variable target_col
    if not isinstance(target_col, str):
        raise TypeError(f"El parámetro target_col debe ser un string")
    # Verificamos el tipo en la variable umbral_corr
    if not isinstance(umbral_corr, float):
        raise TypeError(f"El parámetro umbral_corr debe ser un float")

                        
    # Verificamos si existe algún error al llamar a la columna 'target_col'
    if target_col not in df.columns:
        raise TypeError(f"Error: La columna '{target_col}' no está bien indicada, no se puede asignar como 'target_col'.")


    # Verificamos si 'target_col' es una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        raise TypeError(f"Error: La columna '{target_col}' no es una variable numérica continua.")


    # Verificamos si 'umbral_corr' está en el rango [0, 1]
    if not (0 <= umbral_corr <= 1):
        raise TypeError("Error: Se ha indicado un 'umbral_corr' incorrecto, debe estar entre el rango [0, 1].")

    
    # Verificamos si 'pvalue' es None, float o int y además un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        raise TypeError("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")


    # Obtenemos las columnas numéricas del dataframe
    cols_numericas = df.select_dtypes(include=[np.number]).columns

    # Calculamos la correlación y p-value para cada columna numérica con 'target_col'
    correlaciones = []
    for col in cols_numericas:
        corr, p_value = pearsonr(df[col], df[target_col])
        correlaciones.append((col, corr, p_value))

    # Filtramos las columnas basadas en 'umbral_corr' y 'pvalue'
    features_seleccionadas = []
    for col, corr, p_value in correlaciones:
        #if abs(corr) > umbral_corr and (pvalue is None or p_value < 1 - pvalue):#corregir
        if abs(corr) > umbral_corr and (pvalue is None or p_value < pvalue):#corregido
            features_seleccionadas.append((col, corr, p_value))

    # Devolvemos la lista de características seleccionadas junto con sus correlaciones y valores p
    return features_seleccionadas


def plot_features_num_regression(df:pd.DataFrame,target_col="", columns=list(""), umbral_corr=0.0,pvalue=None):
    """
    Función que devuelve una lista con las columnas numéricas 
    del dataframe cuya correlación con la columna designada 
    por "target_col" sea superior en valor absoluto al valor dado 
    por "umbral_corr". 
    
    Además si la variable "pvalue" es distinta de None, sólo devolvera 
    las columnas numéricas cuya correlación supere el valor indicado y 
    además supere el test de hipótesis con significación 
    mayor o igual a 1-pvalue

    Argumentos:
    
    `df` (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    `target_col` (string): Nombre de la columna target de un modelo de regresión
    
    `columns` (list): Lista con los nombres de las columnas
    
    `umbral_corr` (float): Umbral de correlación, entre 0 y 1. Por defecto valor 0
    
    `pvalue` (int): Valor de significación estadística

    Retorna:

    sns.pairplot: Pairplot
    """
    
    # Verificamos si existe algún error al llamar a la columna 'target_col'
# Verificamos si 'target_col' es una variable numérica continua
    if target_col and not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    
    # Verificamos si 'umbral_corr' está en el rango [0, 1]
    if not (0 <= umbral_corr <= 1):
        print("Error: Se ha indicado un 'umbral_corr' incorrecto, debe estar entre el rango [0, 1].")
        return None
    
    # Verificamos si 'pvalue' es un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        print("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")
        return None
    
    # Si 'target_col' está presente, excluimos esa columna
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col:
        columns.remove(target_col)
    
    # Filtramos las columnas basadas en 'umbral_corr' y 'pvalue'
    selected_columns = []
    for col in columns:
        if target_col:
            corr, p_value = pearsonr(df[col], df[target_col])
            if abs(corr) > umbral_corr and (pvalue is None or p_value < pvalue):#corregido
                selected_columns.append(col)
    
    # Si no hay columnas seleccionadas, mostramos un mensaje y devolvemos None
    if not selected_columns:
        print("No hay columnas que cumplan con los criterios de selección.")
        return None
    
    # Pintamos los pairplots
    sns.pairplot(df[selected_columns + [target_col]])
    
    return selected_columns


def get_features_cat_regression(df:pd.DataFrame,target_col:str="", pvalue=0.05):
    """
    Función que devuelve una lista con las columnas categóricas del 
    dataframe cuyo test de relación con la columna designada por 
    'target_col' supere en confianza estadística el test de relación 
    que sea necesario 
    
    Argumentos:
    
    `df` (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    `target_col` (string): Nombre de la columna target de un modelo de regresión
    
    `pvalue` (float): Valor de significación estadística. Por defecto valor 0.05

    Retorna:

    list: columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario hacer
    """
    
    lista_de_categoricas=list()
    # Verificamos si existe algún error al llamar a la columna 'target_col'
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no está bien indicada, no se puede asignar como 'target_col'.")
        return None
    # Verificamos si 'target_col' es una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None
    # Verificamos si 'pvalue' es un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        print("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")
        return None
    #realizar test de relación de confianza
    #print("Test de correlación")
    df_categoricas=df.drop(target_col,axis=1)
    
    for columna in df_categoricas.columns:
        if not np.issubdtype(df[columna].dtype, np.number):
            if not np.issubdtype(df[columna].dtype, np.datetime64):
                lista_de_categoricas.append(columna)
            #print("lista_de_categoricas:",lista_de_categoricas)

    
    selected_columns = []
    
    for columna in lista_de_categoricas:#averiguo los valores únicos
        unique_values=df[columna].unique()
        parametros="" # para formar la llamada dinámicamente a t-test y chi cuadrado
        for valor in unique_values:
            #parametros=parametros + f'df["{target_col}"][df["{columna}"]=="{valor}"]'
            #if valor != unique_values[len(unique_values)-1]:
            #    parametros=parametros + ","
        
        #print("parametros:",parametros)

        # Create a dictionary to store variable values
        #variables = {}
        # Execute the dynamic code and capture variables in the dictionary
        #exec(f"t_statistic,p_value = f_oneway({parametros})", globals(), variables) #este si
        #exec(instructions,{},variables) #este si
            t_statistic,p_value = f_oneway(df[target_col][df[columna]==valor],df[target_col])
            if p_value<pvalue:#corregido
                if columna not in selected_columns:
                    selected_columns.append(columna)
            else:
                #sentencia=f"chi2_stat, p_value, dof, expected = chi2_contingency(pd.crosstab(df['{columna}'], 
                # df['{target_col}']))"#no es lo correcto u-mann whitney
                #print("sentencia:",sentencia)
                #exec(sentencia, globals(), variables)
                stat, p_value = mannwhitneyu(df[target_col][df[columna]==valor],df[target_col])
                if p_value<pvalue:#corregido
                    if columna not in selected_columns:
                        selected_columns.append(columna)
    
    if not selected_columns:
        print("No hay columnas que cumplan con los criterios de selección.")
        return None      
    else:
        return selected_columns


def plot_features_cat_regression(df:pd.DataFrame,target_col:str="", columns:list=[],pvalue=0.05,with_individual_plot:bool=False):
    """
    Función que pintará los histogramas agrupados de la variable "target_col"
    para cada uno de los valores de las variables categóricas incluidas en 
    columns que cumplan que su test de relación con "target_col" es 
    significativo para el nivel 1-pvalue de significación estadística. 
    La función devolverá los valores de "columns" que cumplan con las 
    condiciones anteriores. 
    
    Argumentos:
    
    df (pandas.DataFrame): Variable que contiene dataframe de Pandas.
    
    target_col (string): Nombre de la columna target de un modelo de regresión
    
    columns(list): Lista con los nombres de las columnas
    
    pvalue (float): Por defecto valor 0.05

    with_indivual_plot(boolean): Por defector valor False

    Retorna:

    sns.pairplot: Pairplot
    """
    # Verificamos si existe algún error al llamar a la columna 'target_col'
    
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no está bien indicada, no se puede asignar como 'target_col'.")
        return None

    # Verificamos si 'target_col' es una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna '{target_col}' no es una variable numérica continua.")
        return None

    # Verificamos si 'pvalue' es un valor válido
    if pvalue is not None and (not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1):
        print("Error: Si 'pvalue' no es 'None', debe tener un valor entre (0, 1).")
        return None
    #print("Entro")
    columnas_cat = []
    for col in columns:
        tabla = pd.crosstab(df[col], df[target_col])
        chi2, p_value_col, _, _ = chi2_contingency(tabla)
        #print("chi2_contingency(tabla)[1] < pvalue",chi2_contingency(tabla)[1])
        #if chi2_contingency(tabla)[1] > pvalue:
        if p_value_col < pvalue:#corregido
            columnas_cat.append(col)
            if with_individual_plot:
                for col in columnas_cat:    
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x=target_col, y=col, data=df)
                    plt.xlabel('Target')
                    plt.ylabel("Col")
                    plt.title("Relación entre target y 'col'")

                    plt.tight_layout()
                    plt.show()
                    
    return columnas_cat             