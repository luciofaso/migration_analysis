import pandas as pd
import numpy as np
import copy
import seaborn as sns

def select_years (df:pd.DataFrame, column_dates:str, years:list):
    """" Select the years from a larger datasests """

    df_out = pd.DataFrame()
    for year in years:
        df_year = df [df[column_dates]== year]
        df_out = df_out.append(df_year,ignore_index=True)
    return(df_out)


def order_by_nan(df:pd.DataFrame):
    """Order dataframe (row and columns) according to presence of data, i.e. more data row-columns on top-left"""

    new_index = df.isnull().sum(1).sort_values().index
    new_col_index = df.isnull().sum().sort_values().index
    df = df.reindex(new_index)
    df = df.reindex(columns=new_col_index)
    return df


def smart_cut(df:pd.DataFrame, threshold=0, weight_col=1):
    """Drop columns and raws with many nan, minimizing the loss of informative data-points
    
    Args: 
        threshold : percentage of nan left in the dataframe
        wheight_col : importance of columns, to penalize dropping of columns more (weight>1) or less (weight<1)
    
    Returns:
        df (pd.DataFrame): clean dataframe 
    """

    while (df.isnull().sum().sum())/(df.shape[0] * df.shape[1]) > threshold:

        worst_row = np.argmax(df.isnull().sum(1))
        worst_row_value = np.max(df.isnull().sum(1))

        worst_col = np.argmax(df.isnull().sum(0))
        worst_col_value = np.max(df.isnull().sum(0))

        # criterium : minimize loss of valid data-points
        if (df.shape[1] - worst_row_value) <= (weight_col*(df.shape[0] - worst_col_value)):
            df = df.drop(worst_row)
        else:
            df = df.drop(worst_col,axis=1)

    return df



def sel_regressor(df:pd.DataFrame, col_target:str):
    """Given a dataframe and the target column, create matrix of regressors X and target y"""
    
    y = df[col_target].values
    X = df.drop([col_target],axis=1).values
    name_inputs = df.drop([col_target],axis=1).columns
    return X,y,name_inputs



def single_factor_plot (name_factor:str, X:np.ndarray, y:np.ndarray, input_names:list, model):
    """ Plot effect of single factor on model output"""

    pos_factor = np.where(input_names==name_factor)[0][0]
    X_factor = X[:,pos_factor]
    X_partial = copy.deepcopy(X)
    X_partial[:,pos_factor] = np.mean(X_factor)
    y_partial = model.predict(X_partial)
    best_score = model.score(X, y)
    loss_score = best_score - model.score(X_partial, y)
    err = y - y_partial
    
    #data preparation
    data4plt =pd.DataFrame(np.transpose(np.array([X_factor,err])),columns=[name_factor,'Effect on model output'])
    
    # plotting
    sns_plt1 = sns.lmplot(data=data4plt,x=name_factor,y='Effect on model output')
    fig = sns_plt1.fig
    fig.suptitle('Influence of single factor, $R^2$: ' + str(loss_score)[0:6])    
