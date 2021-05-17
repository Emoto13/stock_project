import pandas as pd

class PreProcessor:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe
        
    def preprocess_alpha_vantage_df(self):
        self.dataframe['y'] = self.dataframe['close']
        self.dataframe.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        self.dataframe.index.name = 'ds'
        self.dataframe.index = pd.to_datetime(self.dataframe.index)
        self.dataframe.reset_index(inplace=True)
        return self.dataframe
