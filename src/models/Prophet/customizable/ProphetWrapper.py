from .ProphetForecaster import ProphetForecaster

class ProphetWrapper:
        
    def __init__(self, dataframe=None, periodicity="weekly", test_mode=False, **kwargs) -> None:
        super().__init__(dataframe, periodicity, test_mode=test_mode)
        self.model = ProphetForecaster(periodicity=self.periodicity, **kwargs)

    def run(self):
        predictions = self.model.predict(self.dataframe)
        self.model.test()
        return predictions