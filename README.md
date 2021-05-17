# Stock_project

Stock_project is a Python library for stock prediction through different models.

## Current supported models

1. Prophet
2. LSTM
3. Transformer + Time2Vec

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

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip3 install -r requirements.txt
```


## Usage
You can instantiate any of the model wrapper classes or use StockPredictionWrapper for simplicity.

```python
from StockPredictionWrapper import StockPredictionWrapper

key = "your_key"
periodicity = "daily"
symbol = "GOOGL"
    
original_data, forecast = StockPredictionWrapper()\
.run_prediction(symbol=symbol,
                periodicity=periodicity,
                api_key=key,
                convert_to_stationary=True,
                stationarity_test='adf',
                smoother='moving_average',
                save_plot=True,
                model='lstm',
                cutoff=0,
                test_mode=True,
                scale=True,
                changepoint_prior_scale=0.5,
                days_ahead=365,
                window_size=9)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)