from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class PlotWriterReader:
    def __init__(self, original, forecast) -> None:
        self.original = original
        self.forecast = forecast

    def __create_folders(self, save_path='stocks/untitled/plot.png'):
        index_end_directories = save_path.rfind("/")
        dirs = save_path[:index_end_directories]
        Path(dirs).mkdir(mode=0o775, parents=True, exist_ok=True) 

    def save_original_and_prediction_plot(self, save_path='stocks/untitled/double.png'):
        self.__create_folders(save_path=save_path)
        sns.lineplot(x = "ds", y = "y", data = self.original, label="Actual price")
        sns.lineplot(x = "ds", y = "yhat", data = self.forecast, label="Predicted price")
        plt.title('Stock price prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.savefig(save_path)
    
    def save_plot(self, x, y, title="Prediction", xlabel='Date', ylabel='Value', dpi=100, save_path='stocks/untitled/plot.png'):
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)

    def save_prediction_plot(self, save_path='stocks/untitled/plot.png'):
        self.__create_folders(save_path=save_path) 
        self.save_plot(self.forecast['ds'], self.forecast['yhat'], save_path=save_path)

    @staticmethod
    def display_plot(display_path='stocks/untitled/plot.png'):
        pass

#PlotWriterReader.save_plot(x, '/home/emilian/Desktop/stock_project/project/stocks/MSFT/prediction.png' )