import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit


class StockPrices():
    """
    Класс для загрузки DF, первичной подготовки и построения графиков
    """
    
    def __init__(self, 
                 path=None, 
                 data_frame=None):
        """
        Создает экземпляр класса и подгружает таблицу с ценами или делает экземпляр из df
        """
        if path:
            self.df = pd.read_csv(path, parse_dates=["Date"], index_col=["Date"])
        elif data_frame is not None:
            self.df = data_frame
        else:
            print('Используйте метод .parse_yahoo для парсинга данных')

    def parse_yahoo(self, ticket='AAPL', start=None, end=None, interval='1d'):
        """Скачивание данных с yahoo.finance за установленные даты или за последний год"""
        end = end or str(datetime.now().date())

        start = start or str(datetime.now().date() - timedelta(days=365))

        self.df = yf.download(ticket, start=start, end=end, interval=interval)

    def adf_stat_test(self, column_name):
        """
        Проверка стационарности ряда при помощи теста Дикки-Фуллера
        """
        test = adfuller(self.df[column_name])
        print ('adf: ', test[0] )
        print ('p-value: ', test[1])
        print('Critical values: ', test[4])
        if test[0] >  test[4]['5%']: 
            print ('есть единичные корни, ряд не стационарен')
        else:
            print ('единичных корней нет, ряд стационарен')

    def diff_column(self, column_name, periods=1):
        """В DF добавляется столбец первых разностей"""
        new_column_name = '_'.join([column_name, 'diff' + str(periods)])
        self.df[new_column_name] = self.df[column_name].diff(periods=periods)

    def make_features(self):
        """
        Метод добавляет столбцы с признаками из технического анализа
        """
        pass

    def ma(self, column_name, wind):
        """
        Делает столбец со скользящим средним 
        """
        new_column_name = '_'.join([column_name, 'ma' + str(wind)])
        # print(new_column_name)
        self.df[new_column_name] = self.df[column_name].rolling(window=wind).mean()

    def plt_acf(self, column_name, lags=365, save_image=False):
        """
        Построение автокорреляционной зависимости
        """
        plot_acf(self.df[column_name], zero = False, lags = lags)
        plt.grid()
        
        # plt.savefig('doc/fig/price_' + str(year) + '.png', dpi=300)

        plt.show()

    def plt_pacf(self, column_name, lags=20, save_image=False):
        """
        Построение частичной автокорреляционной зависимости
        """
        plot_pacf(self.df[column_name], zero = False, lags = lags)
        plt.grid()

        # if save_image:
        #     plt.savefig('doc/fig/price_' + str(year) + '.png', dpi=300)

        plt.show()

    def plt_candle(self, save_image=False):
        """Свечной график"""
        pass

    def plt_all(self, limits=(), save_image=False):
        pass

    def log_column(self, column_name):
        """Логарифм колонки и добавление результата в df"""
        self.df[column_name + '_log'] = np.log(self.df[column_name])

    def seasonal_decomp(self, column_name, period=5):
        """Декомпозиция ряда"""
        plt.rcParams["figure.figsize"] = 18, 10
        plt.rcParams["font.size"] = 16

        result = seasonal_decompose(self.df[column_name], model='additive', period=period)

        result.plot()
        plt.show()

        return result

    def train_test_split(self, column_name, n_splits=3, test_size=10, gap=0):
        """Проводит разбиение на несколько тренировочных и тестовых наборов"""
        tscv = TimeSeriesSplit(n_splits=3, test_size=test_size, gap=0)

        train_test_groups = tscv.split(self.df[column_name])


# class Make(StockPrices):
#     def prep(self):
#         pass

# if __name__=='main':
#     df_test = StockPrices(path='data/AAPL_20060101-20230131.csv')
#     df_test.head()