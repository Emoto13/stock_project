import pandas as pd

class Preprocessor:
    def __init__(self, dataframe, preprocess='close') -> None:
        self.dataframe = dataframe
        

    @staticmethod
    def preprocess_alpha_vantage_df(dataframe):
        result_df = dataframe.T
        result_df['1. open'] = result_df['1. open'].astype('float')
        result_df['2. high'] = result_df['2. high'].astype('float') 
        result_df['3. low'] = result_df['3. low'].astype('float') 
        result_df['4. close'] = result_df['4. close'].astype('float')
        #result_df['y'] = (result_df['1. open'] + result_df['2. high'] + result_df['3. low'] +result_df['4. close'])/4
        result_df['y'] = result_df['4. close']
        result_df.drop(columns=['1. open', '2. high', '3. low', '4. close', '5. volume'], inplace=True)
        result_df.index.name = 'ds'
        result_df.index = pd.to_datetime(result_df.index)
        print(result_df.info())
        result_df.reset_index(inplace=True)
        return result_df