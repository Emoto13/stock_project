from StockPredictionWrapper import StockPredictionWrapper

key = "your_key"
host = "alpha-vantage.p.rapidapi.com"
function = "TIME_SERIES_WEEKLY"
symbol = "NFLX"
    
prediction_df, forecast = StockPredictionWrapper()\
.run_prediction(url='https://alpha-vantage.p.rapidapi.com/query',
                symbol=symbol,
                function=function,
                rapidapi_key=key,
                rapidapi_host=host,
                convert_to_stationary=True,
                stationarity_test='adf',
                smoother='lowess',
                save_plot=True)
    
