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
