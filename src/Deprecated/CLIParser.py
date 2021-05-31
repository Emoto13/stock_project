import argparse
import pandas as pd
import pandas_datareader as web
from StockPredictionWrapper import StockPredictionWrapper

class CLIParser:
    def __init__(self):
        """ 
        TODO: arguments to add/update:
        1. result_path 
        2. stationarity_test and smoothing
        3. save_plot_path
        """ 
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data_path', type=str, help="Data source path. Pandas supported data source. Currently supported .json, .xlxs, .csv, .pkl")
        self.parser.add_argument('-v', '--version', type=str, default='1', help="Model version to use")
        self.parser.add_argument('-f', '--format', type=str, default='json', help="Export format of model predictions")
        self.parser.add_argument('-r', '--result_path', type=str, default='./results.json', help="Path to export the data")
        self.parser.add_argument('-s', '--symbol', type=str, help="Use symbol flag insted of dpath to get predictions")
        self.parser.add_argument('-m', '--model', type=str, help="Specify which model to use")
        self.parser.add_argument('-p', '--periodicity', type=str, help="Specify periodicity/function")
        self.parser.add_argument('-k', '--api_key', type=str, help="Specify AlphaVantage api key ")
        
        self.parser.add_argument('-st', '--stationarity_test', type=str, help="Specify stationarity test")
        self.parser.add_argument('-sm', '--smoother', type=str, help="Specify smoothing")
        self.parser.add_argument('-spp', '--save_plot_path', type=str, help="Specify path to save plot (actual vs prediction)")
        
                
    def run(self):
        args = self.parser.parse_args()
        prediction_df, forecast = StockPredictionWrapper()\
                .run_prediction(symbol=args.symbol,
                periodicity=args.periodicity,
                api_key=args.api_key,
                convert_to_stationary=True if args.stationarity_test else False,
                stationarity_test=args.stationarity_test,
                smoother=args.smoother,
                save_plot=True if args.save_plot else False,
                save_path=args.save_plot_path,
                model=args.model)
        return prediction_df, forecast


"""
python3 playground.py -f web -s TSLA -m lstm -p weekly -k ca60b9eee5msh87e0638d43a4436p1cb6c9jsn5fe168290094
"""