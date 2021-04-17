from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

class Decomposer:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe
        #decomposer with observed, seasonal, trend and resid fields
        self.decomposed = None
    
    def seasonal_decompose(self, model='additive', field='y', period=None):
        self.decomposed = seasonal_decompose(self.dataframe[field], model=model, extrapolate_trend='freq', period=period)
        return self.decomposed

    @staticmethod
    def detrend(dataframe=None, field='y'):
        detrended = signal.detrend(dataframe[field].values)
        return detrended