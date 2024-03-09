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
