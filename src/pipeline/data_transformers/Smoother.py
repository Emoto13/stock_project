import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

class Smoother:
    def __init__(self) -> None:
        pass

    @staticmethod
    def smooth(signal, type:str ='lowess', window_size=10):
        map_type_to_function = {
            'lowess': Smoother.lowess,
            'moving_average': Smoother.moving_average
        }
        return Smoother.moving_average(signal, window_size)

    @staticmethod
    def lowess(signal, smoothing_percentage=0.05):
        df_lowess = lowess(signal, np.arange(len(signal)), frac=smoothing_percentage)[:, 1]
        print(df_lowess)
        return df_lowess

    @staticmethod
    def moving_average(signal, window_size=10):
        smoothed_signal = pd.Series(signal).rolling(window_size, min_periods=1).mean()
        return smoothed_signal
    
    def lfilter(signal):
        pass
    
    def firwin(signal):
        pass
    
    def savgol_filter(signal):
        pass