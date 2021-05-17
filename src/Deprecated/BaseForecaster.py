from abc import abstractmethod

class BaseForecaster:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self, X_test=None, y_test=None):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass