from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

from .constants import MAP_PERIODICITY_TO_SEASONALITY_PARAMETERS, \
    MAP_PERIODICITY_TO_FREQUENCY, MAP_PERIODICITY_TO_SEASONALITY


class ProphetForecaster:
    def __init__(self, periodicity='weekly', **kwargs):
        """[summary]

        Args:
            periodicity (str, optional): [description]. Defaults to 'weekly'.
            kwargs:
                changepoint_prior_scale (float, optional): [description]
                seasonality_prior_scale (float, optional): [description]
                holidays_prior_scale (float, optional): [description]
                seasonality_mode (str, optional): [description]
        """
        super().__init__(**kwargs)
        self.periodicity = periodicity
        self.model = self.__build_model(periodicity=self.periodicity, **kwargs)
    
    def __get_model_with_seasonality(self, periodicity, **kwargs):
        return Prophet(**MAP_PERIODICITY_TO_SEASONALITY[periodicity], **kwargs)
            
    def __get_seasonality_parameters(self, periodicity):
        return MAP_PERIODICITY_TO_SEASONALITY_PARAMETERS[periodicity]
    
    def __get_frequency_parameters(self, periodicity):
        return MAP_PERIODICITY_TO_FREQUENCY[periodicity]
    
    def __build_model(self, periodicity='weekly', **kwargs):
        model = self.__get_model_with_seasonality(periodicity, **kwargs)
        seasonality_params = self.__get_seasonality_parameters(periodicity)        
        model.add_seasonality(
            name=periodicity, 
            period=seasonality_params['period'],
            fourier_order=seasonality_params['fourier_order']
        )
        return model

    def train(self):
        pass

    def test(self):
        df_cv = cross_validation(self.model, horizon = '365 days')
        df = performance_metrics(df_cv)
        df.drop(['horizon'], axis=1, inplace=True)
        df.apply(np.sum, inplace=True)
        print(df)
        return df
                    
    def predict(self, X=None):
        self.model.fit(X)
        future_dataframe_params = self.__get_frequency_parameters(self.periodicity)
        future_df = self.model.make_future_dataframe(periods=future_dataframe_params['periods'], freq=future_dataframe_params['freq'])
        predictions = self.model.predict(future_df)
        predictions = predictions[['ds', 'yhat']]
        predictions.set_index('ds', inplace=True)
        return predictions