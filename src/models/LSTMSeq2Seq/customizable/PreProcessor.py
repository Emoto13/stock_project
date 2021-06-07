import numpy as np

class PreProcessor:
  @staticmethod
  def prepare_data(timeseries, n_steps_back, n_steps_ahead):
    X, y = [], []
    for i in range(n_steps_back, len(timeseries) - n_steps_ahead + 1):
      X.append(timeseries[i-n_steps_back:i])
      y.append([timeseries[i:i+n_steps_ahead]]*n_steps_ahead)
    return np.array(X), np.array(y)