import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import ast
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind,mannwhitneyu,ttest_rel,ttest_1samp
from scipy.stats import chi2_contingency,f_oneway

def internal_anova_columns (df:pd.DataFrame,target_col:str,lista_num_columnas:list=[],pvalue:float=0.05):
    """
    Uso interno, devuelve una lista con las columnas del dataframe cuyo ANOVA con la columna designada por "target_col" 
    supere el test de hipótesis con significación mayor o igual a 1-pvalue

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `pvalue` (float): Variable float con valor por defecto 0.05

    Retorna:
    col_nums: List
    """
    selected_columns=[]
    for columna in lista_num_columnas:
        unique_values=df[columna].unique()
        for valor in unique_values:
            t_statistic,p_value = f_oneway(df[target_col][df[columna]==valor],df[target_col])
            if p_value<pvalue:#corregido
                if columna not in selected_columns:
                    selected_columns.append(columna)
    return selected_columns

def get_features_num_classification (df:pd.DataFrame,target_col:str,pvalue:float=0.05):
    """
    La función devuelve una lista con las columnas numéricas del dataframe cuyo ANOVA con la columna designada por "target_col" 
    supere el test de hipótesis con significación mayor o igual a 1-pvalue

    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `pvalue` (float): Variable float con valor por defecto 0.05.

    Retorna:
    col_nums: List
    """
    #Validaciones
    
    if target_col=="":
        raise ValueError("El parámetro target_col no puede estar vacío")
    if target_col not in df.columns:
        raise ValueError("El parámetro target_col debe estar en el DataFrame")
    #Búsqueda de columnas numéricas
    num_values=['float32','int32','float64','int64','int8', 'int16', 'float16','uint8', 'uint16', 'uint32', 'uint64']
    lista_num_columnas=[]
    for columna in df.columns:
        if columna.dtype in num_values:
            lista_num_columnas.append(columna)
    #ANOVA
    selected_columns=[]
    selected_columns= internal_anova_columns(df,target_col,lista_num_columnas,pvalue)
    '''
    for columna in lista_num_columnas:
        unique_values=df[columna].unique()
        for valor in unique_values:
            t_statistic,p_value = f_oneway(df[target_col][df[columna]==valor],df[target_col])
            if p_value<pvalue:#corregido
                if columna not in selected_columns:
                    selected_columns.append(columna)
    '''
    return selected_columns

def plot_features_num_classification (df:pd.DataFrame,target_col:str="",columns:list=[],pvalue:float=0.5):
    """
    La función pinta una pairplot del DataFrame considerando la columna designada por "target_col" y aquellas incluidas en "column" que cumplan el test de ANOVA 
    para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 
    Se espera que las columnas sean numéricas. El pairplot utiliza como argumento de hue el valor de target_col
    Si target_Col es superior a 5, se usan diferentes pairplot diferentes, se pinta un pairplot por cada 5 valores de target posibles.
    Si la lista de columnas a pintar es grande se pinten varios pairplot con un máximo de cinco columnas en cada pairplot,
    siendo siempre una de ellas la indicada por "target_col"
    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `columns` (list): Variable con la lista de columnas de tipo list.
    `pvalue` (float): Variable float con valor por defecto 0.5.    
    
    Retorna:
    sns.pairplot: Pairplot
    """
    selected_columns= internal_anova_columns(df,target_col,columns,pvalue)
    if len(selected_columns)>0:
        if len(selected_columns)>5:
            #codigo para bloques de 5
            pass
        else:
            #unico
            pass
    return None

#Javier
def get_features_cat_classification(df:pd.DataFrame, target_col:str, mi_threshold:float=0.0, normalize:bool=False):
    '''
    Esta función recibe como argumentos un dataframe, el nombre de una de las columnas del mismo (argumento 'target_col'), que debería ser el target de un hipotético 
    modelo de clasificación, es decir debe ser una variable categórica o numérica discreta pero con baja cardinalidad, un argumento "normalize" con valor False por defecto, 
    una variable float "mi_threshold" cuyo valor por defecto será 0.
    * En caso de que "normalize" sea False:
        La función debe devolver una lista con las columnas categóricas del dataframe cuyo valor de mutual information con 'target_col' iguale o supere 
        el valor de "mi_threshold".
    * En caso de que "relative" sea True:
        La función debe devolver una lista con las columnas categóricas del dataframe cuyo valor normalizado de mutual information con 'target_col' iguale o supere 
        el valor de "mi_threshold". 
        El valor normalizado de mutual information se considera el obtenido de dividir el valor de mutual information tal cual ofrece sklearn o la fórmula de cálculo 
        por la suma de todos los valores de mutual information de las features categóricas del dataframe.
    En este caso, la función debe comprobar que "mi_threshold" es un valor float entre 0 y 1, y arrojar un error si no lo es.
    La función debe hacer todas las comprobaciones necesarias para no dar error como consecuecia de los valores de entrada.
    Es decir hará un check de los valores asignados a los argumentos de entrada y si estos no son adecuados debe retornar None y printar 
    por pantalla la razón de este comportamiento. Ojo entre las comprobaciones debe estar que "target_col" hace referencia a una variable numérica continua del dataframe.
    
    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `mi_threshold` (float): Variable float con valor por defecto 0.0.
    `normalize` (bool): Variable bool con valor por defecto False.
    
    Retorna:
    selected_columns: List

    '''
    # Comprobación de argumentos de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame de pandas.")
        return None
    
    if not isinstance(target_col, str):
        print("Error: 'target_col' debe ser una cadena de caracteres.")
        return None
    
    if target_col not in df.columns:
        print("Error: 'target_col' no está presente en el DataFrame.")
        return None
    
    if df[target_col].dtype not in ['object', 'category']:
        print("Error: 'target_col' debe ser una variable categórica o numérica discreta del DataFrame.")
        return None
    
    if not isinstance(mi_threshold, float):
        print("Error: 'mi_threshold' debe ser un valor float.")
        return None
    
    if not 0 <= mi_threshold <= 1 and normalize:
        print("Error: 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None
    
def plot_features_cat_classification(df:pd.DataFrame, target_col:str="", columns:list=[], mi_threshold:float=0.0, normalize:bool=False):
    """
    La función pinta una pairplot del DataFrame considerando la columna designada por "target_col" y aquellas incluidas en "column" que cumplan el test de ANOVA 
    para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 
    Se espera que las columnas sean numéricas. El pairplot utiliza como argumento de hue el valor de target_col
    Si target_Col es superior a 5, se usan diferentes pairplot diferentes, se pinta un pairplot por cada 5 valores de target posibles.
    Si la lista de columnas a pintar es grande se pinten varios pairplot con un máximo de cinco columnas en cada pairplot,
    siendo siempre una de ellas la indicada por "target_col"
    Argumentos:

    `df` (DataFrame): Variable dataframe de Pandas.
    `target_col` (str): Variable target tipo str.
    `columns` (list): Variable con la lista de columnas de tipo list.
    `pvalue` (float): Variable float con valor por defecto 0.5.    
    
    Retorna:
    sns.pairplot: Pairplot
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame de pandas.")
        return
    
    if not isinstance(target_col, str):
        print("Error: 'target_col' debe ser una cadena de caracteres.")
        return
    
    if target_col and df[target_col].dtype not in ['object', 'category']:
        print("Error: 'target_col' debe ser una variable categórica del DataFrame.")
        return
    
    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista de strings.")
        return
    
    if not 0 <= mi_threshold <= 1 and normalize:
        print("Error: 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return
    
    if not columns:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    else:
        categorical_cols = columns
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