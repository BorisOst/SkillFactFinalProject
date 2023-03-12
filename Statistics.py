import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Statistics():
    """Классс для работы со статистиками и ошибками """

    def __init__(self, series_act, series_pred):
        """
        series_act: actuals - реальные значения временного ряда
        series_pred: predictions - предсказанные значения временного ряда
        """
        self.actuals = np.array(series_act)
        self.predictions = np.array(series_pred)

    def mae(self):
        mae = mean_absolute_error(self.actuals, self.predictions)
        return mae

    def mse(self):
        mse = mean_squared_error(self.actuals, self.predictions)
        return mse

    def report(self):
        print(f"MAE: {self.mae:.2f}")
        print(f"MSE: {self.mse:.2f}")
        print(f"MRSE: {np.sqrt(self.mse):.2f}")
        