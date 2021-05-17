from .Prophet import ProphetWrapper
from .LSTM import LSTMWrapper
from .Transformer import TransformerWrapper

map_name_to_model = {
    'prophet': ProphetWrapper,
    'lstm': LSTMWrapper,
    'transformer': TransformerWrapper
}

class ModelSelector:
    def __init__(self, pred_df, model='prophet', periodicity='weekly', **kwargs):
        self.model = map_name_to_model[model](pred_df, periodicity=periodicity, **kwargs)
    
    def run(self):
        return self.model.run()