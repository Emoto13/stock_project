from visualization.PlotWriterReader import PlotWriterReader
from pipeline.data_transformers import PreProcessor, Smoother
from models import ModelSelector
from utils import StationarityTester, DateTimeOperator
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web

class StockPredictionWrapper:
    def __init__(self) -> None:
        self.dataframe = None
        self.forecast = None

    def __convert_time_series_to_stationary(self, dataframe,
                                            stationarity_test: str = 'kdss',
                                            smoother='moving_average',
                                            window_size=10):
        if not StationarityTester.run_test(dataframe, stationarity_test):
            return Smoother.smooth(dataframe, smoother, window_size)
        return dataframe

    def run_prediction(self,
                       api_key: str,
                       symbol: str = "NFLX",
                       periodicity: str = "weekly",
                       model: str = "prophet",
                       test_mode: bool = True,
                       convert_to_stationary: bool = False,
                       stationarity_test: str = 'kdss',
                       smoother: str = 'lowess',
                       scale: bool = True,
                       cutoff: int = 0,
                       save_plot: bool = False,
                       save_path: str = None,
                       window_size: int = 10,
                       **kwargs):
        """
            Runs Prediction for a given stock

            url (str): To communicate with the external API

            symbol (str): standard stock symbol

            function (str): choose stock time periods -> (options):
            'TIME_SERIES_MONTHLY',
            'TIME_SERIES_WEEKLY',
            'TIME_SERIES_DAILY' etc.
            for more info:
            https://rapidapi.com/alphavantage/api/alpha-vantage?endpoint=apiendpoint_9af393c9-35ea-4cc5-b146-1bf11049e5c7

            datatype (str): 'json' (default, recommended),
            other formats might not be supported

            convert_to_stationary (bool): if dataframe is non-stationary,
            it will be converted to stationary

            stationary_test (str) (options): 'kdss', 'adf'

            smoother (str): convert non-stationary series to stationary
            -> (options): loess, lowess, moving_average
        """

        df = web.DataReader(symbol, f"av-{periodicity}", api_key=api_key)
        pred_df = PreProcessor(df).preprocess_alpha_vantage_df()
        self.dataframe = pred_df.copy()
        
        if scale:
            scaler = MinMaxScaler(feature_range=(0,1))
            pred_df.y = scaler.fit_transform(pred_df.y.values.reshape(-1,1))[:,0]
        
        if convert_to_stationary:
            pred_df.y = self.__convert_time_series_to_stationary(pred_df.y.values,
                            stationarity_test=stationarity_test,
                            smoother=smoother,
                            window_size=window_size)
        if cutoff > 0:
            pred_df = pred_df[cutoff:]
            
        forecast = ModelSelector(pred_df, model=model, periodicity=periodicity, test_mode=test_mode, **kwargs).run()
        
        save_path = save_path if save_path else \
        f'reports/stocks/predictions/{symbol}/{DateTimeOperator.get_current_date_and_time()}_{model}_{periodicity}.png'
        
        if scale:
            forecast.yhat = scaler.inverse_transform(forecast.yhat.values.reshape(-1,1))[:,0]
        
        if save_plot:
            PlotWriterReader(self.dataframe, forecast).save_original_and_prediction_plot(save_path=save_path)

        self.forecast = forecast
        return self.dataframe, self.forecast
