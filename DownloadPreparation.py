import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


class StockPrices():
    """
    Класс для загрузки DF, первичной подготовки и построения графиков
    """
    def __init__(self, 
                 path=None, 
                 stock=None, 
                 start=None, 
                 end=None, 
                 interval=None):
        """
        Создает экземпляр класса 
        """
        self.__check_data__(path, stock, start, end, interval)

        if path:
            self.df = self.download(path)

        else:
            print('Используйте метод класса .parse для загрузки цен акций с yahoo finance.')
            print('Параметры метода: ')

    def download(self, path):
        df = pd.read_csv(path)
        return df

    def parse(self):
        df = yf.download()

    def plot_candle(self):
        pass

    def plot_all(self, limits=()):
        pass

    def __check_data__(self):
        pass