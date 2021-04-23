import pandas as pd

class RawDataTransformer:
    def __init__(self, raw_data=None) -> None:
        self.raw_data = raw_data

    # Specify which fields to use to create df.
    # If the field(s) is nested pass an array of fields 
    # and it will create the df correctly
    def __go_deeper(self, data, current_key, key):
        if current_key == key:
            return data[key]

        if type(data[current_key]) is dict:
            return self.__find_value(data[current_key], key)

    def __find_value(self, data, key_to_find):
        for key in data:
            res = self.__go_deeper(data, key, key_to_find)
            if res:
                return res
    
    def to_dataframe(self, field=None):
        data = self.raw_data
        if field:
            data = self.__find_value(data, field)
        self.dataframe = pd.DataFrame(data)
        return self.dataframe
    


#dw = DataWrangler(StockDataFetcher.fetch_data_alpha_vantage())
#dw.to_dataframe(['Monthly Time Series'])
#dw.to_dataframe(['Weekly Time Series'])
#dw.to_dataframe(['Time Series (Daily)'])
#print(dw.dataframe)