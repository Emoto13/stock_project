import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

class Smoother:
    def __init__(self) -> None:
        pass

    @staticmethod
    def smooth(dataframe, type='lowess', field='y'):
        map_type_to_function = {
            'lowess': Smoother.lowess,
            'moving_average': Smoother.moving_average
        }
    
        return map_type_to_function[type](dataframe, field=field)

    @staticmethod
    def lowess(dataframe, smoothing_percentage=0.05, field='y'):
        df_lowess = pd.DataFrame(lowess(dataframe[field], np.arange(len(dataframe[field])),
        frac=smoothing_percentage)[:, 1], index=dataframe.index, columns=[field])
        dataframe['y'] = df_lowess
        return dataframe

    @staticmethod
    def moving_average(dataframe, field='y'):
        df_ma = dataframe[field].rolling(3, center=True, closed='both').mean()
        dataframe[field] = df_ma.iloc[1:]
        print(df_ma)
        return dataframe