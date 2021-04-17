import pandas as pd

class DataTransformer:
    def __init__(self, raw_data=None) -> None:
        self.raw_data = raw_data

    # Specify which fields to use to create df.
    # If the field(s) is nested pass an array of fields 
    # and it will create the df correctly
    def to_dataframe(self, fields_to_use=None):
        data = self.raw_data

        if fields_to_use:
            for field in fields_to_use:
                data = data[field]

        self.dataframe = pd.DataFrame(data)
        return self.dataframe

#dw = DataWrangler(StockDataFetcher.fetch_data_alpha_vantage())
#dw.to_dataframe(['Monthly Time Series'])
#dw.to_dataframe(['Weekly Time Series'])
#dw.to_dataframe(['Time Series (Daily)'])
#print(dw.dataframe)