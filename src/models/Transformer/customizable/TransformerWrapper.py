from .TransformerForecaster import TransformerForecaster
from .PreProcessor import PreProcessor
from .constants import MODEL_CHECKPOINT, SEQ_LEN
class TransformerWrapper:
    def __init__(self, dataframe=None, periodicity='weekly', test_mode=False, days_ahead=30, load=False, load_path=MODEL_CHECKPOINT, **kwargs):
        super().__init__(dataframe=dataframe, periodicity=periodicity, test_mode=test_mode)
        self.load = load
        self.days_ahead = days_ahead
        self.model = TransformerForecaster()
        self.X, self.y = PreProcessor.prepare_data(self.dataframe.y.values, SEQ_LEN)
        self.X_train, self.y_train, self.X_test, self.y_test = PreProcessor.split(self.X, self.y)

    def train_and_test(self):
        self.model.train(self.X_train, self.y_train)
        # Use best model 
        self.model.load_weights()
        if self.test_mode:
            self.model.test(self.X_test, self.y_test)
    
    def predict(self, X=None):
        #print('here', X.shape, self.X_train_c.shape) # here (2737,) (2436, 30)
        full = self.model.predict(X)
        train = self.model.predict(self.X_train)
        test = self.model.predict(self.X_test)
        return full, train, test

    def run(self):
        if not self.load:
            self.train_and_test()
        full, train, test = self.predict(self.X, days_ahead=self.days_ahead)
        return full, train, test
