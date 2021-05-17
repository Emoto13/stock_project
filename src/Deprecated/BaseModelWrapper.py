from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    def __init__(self, dataframe=None, periodicity='weekly', test_mode=False, **kwargs):
        self.dataframe = dataframe
        self.periodicity = periodicity
        self.test_mode = test_mode
        
    @abstractmethod
    def run(self):
        pass
        
    