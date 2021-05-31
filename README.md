# Stock_project

Stock_project is a Python library for stock prediction through different models.

## Current supported models

1. [Prophet](https://github.com/Emoto13/stock_project/blob/main/src/models/Prophet/README.md)
2. [LSTM](https://github.com/Emoto13/stock_project/blob/main/src/models/LSTM/README.md)
3. [Transformer + Time2Vec](https://github.com/Emoto13/stock_project/blob/main/src/models/Transformer/README.md)

To be added:
4. Bidirectional LSTM
5. LSTM + Time2Vec
6. Bidirectional LSTM + Time2Vec
7. GRU
8. Bidirectional GRU
9. Capsule Network
10. Graph NNs
11. ARIMA


## Recommended installation and usage

```bash 
git clone github.com/Emoto13/stock_project
```

Reconfigure main.py file with [RAPIDAPI](https://rapidapi.com/) key and tune StockPredictionWrapper instance.
Then the following command will take care if installing all needed packages and running main.
 
```bash
make run_main
```

## Alternative Installation

Use the package manager [pip3](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip3 install -r requirements.txt
```


## Usage
You can instantiate any of the <model-name>Wrapper class for stock prediction. Refer to the documentation of each model for details.

[Example usage](https://github.com/Emoto13/stock_project/blob/main/src/main.py):

```python
from visualization.PlotWriterReader import PlotWriterReader
from pipeline.data_transformers import PreProcessor, Smoother
from utils import DateTimeOperator
from models import LSTMWrapper
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web

key = "key"
periodicity = "daily"
symbol = "GOOGL"

df = web.DataReader(symbol, f"av-{periodicity}", api_key=key)
pred_df = PreProcessor(df).preprocess_alpha_vantage_df()
original_df = pred_df.copy()
original_df.set_index(['ds'], inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))
pred_df.y = scaler.fit_transform(pred_df.y.values.reshape(-1,1))[:,0]

pred_df.y = Smoother.smooth(pred_df.y.values, 'moving_average', 30)
    
save_path = f'reports/stocks/predictions/ \
              {symbol}/{DateTimeOperator.get_current_date_and_time()}_lstm_{periodicity}.png'
lstmw = LSTMWrapper(dataframe=df, periodicity='daily',
                    epochs=35, neurons=10, dropout=0.2, days_ahead=300)
forecast = lstmw.run()


pred_df.rename(columns={"y":"yhat"}, inplace=True)
result = pred_df.append(forecast)
result.set_index(['ds'], inplace=True)
result.yhat = scaler.inverse_transform(result.yhat.values.reshape(-1,1))[:,0]

PlotWriterReader(original_df, result).save_original_and_prediction_plot(save_path=save_path)


```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)