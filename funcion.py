# %%
#importacion de librerias
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd

def limpiar_Dframe(business, economy):
    #carga de librerias dentro de la funcion
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.feature_selection import SequentialFeatureSelector

     # CARGA DE DATOS  
    df1 = pd.read_excel('business.xlsx')
    df2 = pd.read_excel('economy.xlsx')
    pd.set_option('display.max_columns', None)
    print(f"    Titulos de las columnas df1\n{df1.columns}\n{df1.describe()}\n{df1.dtypes}\nVacios {df1.isnull().sum()}")
    print(f"    Titulos de las columnas df2\n{df2.columns}\n{df2.describe()}\n{df2.dtypes}\nVacios {df2.isnull().sum()}")
    df1["class"] = 1
    df2["class"] = 0

    # COMBINACION DE DATOS
    df = pd.concat([df1, df2], ignore_index=True)

    #boxplot
    plt.figure(figsize=(20, 5))
    df.boxplot()
    plt.xticks(rotation=90)
    plt.title('Boxplot de las características numéricas')
    plt.show() 

    #depura y elimina las las ciudades de parada
    df['stop'] = df['stop'].str.replace(r'[\n\tVia]', '', regex=True
                                        )
    #depura y elimina las las ciudades de parada DESPUES DEL ESPACIO
    df['stop'] = df['stop'].str.split().str[0]

    #se convierte la columna 'stop' 'airline', 'class' 'from' y 'to' en variables dummy
    df = pd.get_dummies( df, columns=['stop'] )
    df = pd.concat([df, pd.get_dummies(df['from'], prefix='from')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['to'], prefix='to')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['airline'], prefix='airline')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['class'], prefix='class')], axis=1)
    df.rename(columns={'class_0': 'economy', 'class_1': 'business'}, inplace=True)

    #se pasa la columna de time_taken a minutos
    df['time_taken'] = df['time_taken'].str.replace(r'[hm]', '', regex=True)
    df[['hora', 'minuto']] = df['time_taken'].str.split(' ', expand=True)
    df['hora'] = pd.to_numeric( df['hora'] )
    df['minuto'] = pd.to_numeric( df['minuto'] )
    df['total_time_taken'] = df['hora']*60 + df['minuto']

    #guardar el dframe limpio y categorizado
    dfLimpio = df.copy()
    dfLimpio = df.drop(columns=['num_code', 'arr_time', 'class', 'time_taken', 'from', 'to', 'airline', 'dep_time', 'ch_code', 'date', 'hora', 'minuto'])

    #se elimina celdas de price que tienen comas, ya que no se pueden convertir a numerico y son pocas
    dfLimpio = dfLimpio [~dfLimpio['price'].astype(str).str.contains(',', na=False)]

    #se eliminan los valores nulos 
    dfLimpio = dfLimpio.dropna()
    dfLimpio.isnull().sum()

    #convertir price a numerico por ser la columna objetivo
    dfLimpio['price'] = pd.to_numeric( dfLimpio['price'] )
    print("ANALISIS UNIVARIADO")
    print(dfLimpio.describe())

    # Histogramas
    dfLimpio.hist(bins=30, figsize=(15, 4))
    plt.tight_layout()
    plt.show()

    # Boxplots
    plt.figure(figsize=(12, 4))
    dfLimpio.boxplot()
    plt.xticks(rotation=90)
    plt.show()

    # Price vs otras variables (scatter plots)
    print("Analisis bivariado")
    columnas_num = dfLimpio.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 3
    n_rows = len(columnas_num) // n_cols + 1

    plt.figure(figsize=(15, 5*n_rows))
    for i, col in enumerate(columnas_num, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.scatterplot(x=col, y='price', data=dfLimpio)
        plt.title(f'Price vs {col}')
    plt.tight_layout()
    plt.show()

    # lista de correlaciones
    correlaciones = dfLimpio.corr()['price'].sort_values(ascending=False)
    print("TOP correlaciones con Price:")
    print(correlaciones.head(14))

    X = dfLimpio.drop(columns=['price'])
    y = dfLimpio['price']

    #se rrellenan los valores nulos con la media de cada columna
    X.fillna(X.mean(), inplace=True)
    lr = LinearRegression()
    sfs = SequentialFeatureSelector(lr, n_features_to_select=25, direction='forward')
    sfs.fit(X, y)
    Variables = X.columns[sfs.get_support()]
    print("Variables seleccionadas metodo FORWARD SELECTION:", Variables)

    #Matriz de correlacion
    corr = dfLimpio.corr()
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()
    
    #dframe para modelos seleccionado de FORWARD SELECTION
    dfModel = dfLimpio[Variables.tolist() + ['price']]

    return dfModel
