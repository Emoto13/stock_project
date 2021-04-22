from visualization.PlotWriterReader import PlotWriterReader
from pipeline.data_transformers import Preprocessor, Smoother, DataTransformer
from pipeline.data_fetchers import StockDataFetcher
from models import Predictor
from utils import StationarityTester

map_function_to_field_name = {
    'TIME_SERIES_MONTHLY': ['Monthly Time Series'],
    'TIME_SERIES_WEEKLY': ['Weekly Time Series'],
    'TIME_SERIES_DAILY': ['Time Series (Daily)']
}


class StockPredictionWrapper:
    def __init__(self) -> None:
        self.dataframe = None
        self.forecast = None

    def __convert_time_series_to_stationary(self, dataframe,
                                            stationarity_test: str = 'kdss',
                                            smoother='moving_average'):
        if not StationarityTester.run_test(dataframe, stationarity_test):
            return Smoother.smooth(dataframe, smoother)
        return dataframe

    def run_prediction(self,
                       rapidapi_key: str,
                       rapidapi_host: str,
                       url: str = 'https://alpha-vantage.p.rapidapi.com/query',
                       symbol: str = "NFLX",
                       function: str = "TIME_SERIES_WEEKLY",
                       convert_to_stationary: bool = False,
                       stationarity_test: str = 'kdss',
                       smoother: str = 'lowess',
                       cutoff: int = 100,
                       save_plot: bool = False,
                       save_path: str = 'plot.png'):
        '''
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
        '''
        querystring = {"symbol": symbol,
                       "function": function,
                       "outputsize": "compact",
                       "datatype": "json"}
                
        raw_data = StockDataFetcher(url=url,
                                    querystring=querystring,
                                    rapidapi_key=rapidapi_key,
                                    rapidapi_host=rapidapi_host)\
        .fetch_data()

        field_to_use = map_function_to_field_name[function]
        df = DataTransformer(raw_data).to_dataframe(field_to_use)
        pred_df = Preprocessor(df).preprocess_alpha_vantage_df()

        if convert_to_stationary:
            pred_df = self.__convert_time_series_to_stationary(pred_df,
                            stationarity_test=stationarity_test,
                            smoother=smoother)

        save_path = save_path if save_path != 'plot.png' else f'reports/stocks/predictions/{symbol}/{function}_at_today.png'
        forecast = Predictor.predict(pred_df[:-cutoff], model='prophet')

        if save_plot:
            PlotWriterReader(pred_df, forecast).save_original_and_prediction_plot(save_path=save_path)
        self.dataframe, self.forecast = pred_df, forecast
        return pred_df, forecast

#x, y = StockPredictionWrapper().run_prediction(function="TIME_SERIES_WEEKLY",
#symbol='NFLX',
#convert_to_stationary=True,
#stationarity_test='adf',
#smoother='lowess',
#save_plot=True)