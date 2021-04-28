from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

import numpy as np
import time


class Decomposer:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe
        #decomposer with observed, seasonal, trend and resid fields
        self.decomposed = None
    
    def seasonal_decompose(self, model='additive', field='y', period=None):
        self.decomposed = seasonal_decompose(self.dataframe[field], model=model, extrapolate_trend='freq', period=period)
        return self.decomposed
    
    @staticmethod
    def series_seasonal_decompose(series, model='additive',period=None):
        decomposed = seasonal_decompose(series, model=model, extrapolate_trend='freq', period=period)
        return decomposed
    
    @staticmethod
    def get_periodicty_fft(signal):
        """[summary]

        Returns:
            [type]: [description]
        """
        print(signal.size, len(signal))
        fourier = np.fft.fft(signal)
        freq = np.fft.fftfreq(signal.size)
        print(fourier[:10], freq[:10])
        return 
        
    @staticmethod
    def detrend(series):
        detrended = signal.detrend(series.values)
        return detrended