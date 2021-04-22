from prophet import Prophet
import matplotlib.pyplot as plt
from pathlib import Path

map_query_to_seasonality = {
    'TIME_SERIES_MONTHLY': 'monthly',
    'TIME_SERIES_WEEKLY': 'weekly',
    
}


class Predictor:
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def save_forecast_plot(model, forecast, save_path=''):
        index_end_directories = save_path.rfind("/")
        dirs = save_path[:index_end_directories]
        Path(dirs).mkdir(mode=0o775, parents=True, exist_ok=True) 
        model.plot(forecast)
        plt.savefig(save_path)

    @staticmethod
    def predict(dataframe, model='prophet'):
        map_model_to_function = {
            'prophet': Predictor.predict_prophet
        }
        return map_model_to_function[model](dataframe)

    @staticmethod
    def predict_prophet(dataframe, seasonality='weekly'):
        model = Prophet()
        model.add_seasonality(
        name=seasonality, 
        period=30.5, 
        fourier_order=5
        )
        model.fit(dataframe)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat']]
        forecast.set_index('ds', inplace=True)
        return forecast
