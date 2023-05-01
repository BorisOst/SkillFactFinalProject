import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


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
        plt.rcParams["figure.figsize"] = 18, 10
        plt.rcParams["font.size"] = 20

        plot_acf(self.df[column_name], zero = False, lags = lags)
        plt.grid()
        
        if save_image:
            image_name = 'doc/fig/acf_' + str(column_name) + str(lags) + '.png'
            plt.savefig(image_name, dpi=dpi)

        plt.show()

    def plt_pacf(self, column_name, lags=20, save_image=False, dpi=300):
        """
        Построение частичной автокорреляционной зависимости
        """
        plt.rcParams["figure.figsize"] = 18, 10
        plt.rcParams["font.size"] = 20

        plot_pacf(self.df[column_name], zero = False, lags = lags)
        plt.grid()

        if save_image:
            image_name = 'doc/fig/pacf_' + str(column_name) + str(lags) + '.png'
            plt.savefig(image_name, dpi=dpi)

        plt.show()

    def plt_price_ma(self, column_name, ma_list=(10, 30), save_image=False):
        """Построение группы графиков цена + движущегося среднего"""

        plt.rcParams["figure.figsize"] = 18, 5 * len(ma_list)
        plt.rcParams["font.size"] = 14
        
        fig, ax = plt.subplots(len(ma_list), 2)

        for i in range(len(ma_list)):
            ax[i, 0].plot(self.df[column_name], 
                          color = "blue", 
                          label = column_name + "_Price")
            ax[i, 0].set_title(f"{column_name} Price", size = 24)
            ax[i, 0].grid()
            ax[i, 0].legend()

            ax[i, 1].plot(self.df[column_name].rolling(window=ma_list[i]).mean(), 
                          color = "orange", 
                          label = f"Smoothed {column_name} Prices")
            ax[i, 1].set_title(f"Moving Average ({ma_list[i]}) {column_name} Prices", 
                               size = 24)
            ax[i, 1].grid()
            ax[i, 1].legend()

        plt.show()

    @staticmethod
    def plt_mas(data_series, ma_list, graph_name, save_image=False, dpi=300):
        """
        Столбец + несколько MA на одном графике. mas - это типа moving average во множественном числе
        """
        plt.rcParams["figure.figsize"] = 18, 10
        plt.rcParams["font.size"] = 20

        fig = plt.figure()

        plt.plot(data_series, label=graph_name)

        for ma in ma_list:
            plt.plot(data_series.rolling(window=ma).mean(), label='MA'+str(ma))

        plt.title(f"{graph_name}", size=24)
        plt.grid()
        plt.legend()

        if save_image:
            image_name = 'doc/fig/MAs_' + str(ma_list)[1:-1] + '.png'
            plt.savefig(image_name, dpi=dpi)

        plt.show()

    def plt_candle(self, save_image=False):
        """Свечной график"""
        pass

    def plt_price(self, column_name, save_image=False):
        """График столбца"""
        plt.rcParams["figure.figsize"] = 12, 4

        plt.plot(self.df[column_name])
        plt.title(column_name, size=20)
        plt.grid()
        plt.show()

    def plt_all(self, limits=(), save_image=False):
        pass

    # @staticmethod
    # def save_image(image_name, dpi=300):
    #     if image_name:
    #         plt.savefig(image_name, dpi=dpi)

    def log_column(self, column_name):
        """Логарифм колонки и добавление результата в df"""
        self.df[column_name + '_log'] = np.log(self.df[column_name])

    def seasonal_decomp(self, column_name, period=5):
        """Декомпозиция ряда"""
        result = seasonal_decompose(self.df[column_name], model='additive', period=period)

        plt.rcParams["figure.figsize"] = 18, 10
        plt.rcParams["font.size"] = 16

        result.plot()

        plt.show()

        return result

    def train_test_split(self, column_name, n_splits=3, test_size=10, gap=0):
        """Проводит множественное разбиение на несколько тренировочных и тестовых наборов"""
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        self.train_test_groups = tscv.split(self.df[column_name])

        # train_test_dict_tmp = dict()

        # for train_index, test_index in train_test_groups:
        #     pass


class ARIMAclsModels(StockPrices):
    """Класс для работы с моделями ARIMA и SARIMA"""
    def prep(self):
        print('prep')

    @staticmethod
    def best_arima(train_df, 
                   start_p=1, start_q=1, test='adf',
                   max_p=5, max_q=5, m=5, 
                   start_P=0, start_Q=0, seasonal=True,
                   max_P=4, max_D=4, max_Q=4, 
                   d=1, D=1, trace=True,
                   error_action='ignore',
                   suppress_warnings=True, 
                   stepwise=True):
        """Подбор оптимальных параметров модели"""
        pass

    @staticmethod
    def arima_predict_df(df_train, df_test, pdq=(1, 1, 1)):
        """Формирует DF с предсказанием по ARIMA и доверительным интервалом"""
        # Отключение предупреждений
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")
            
            arima_model = ARIMA(df_train, order=pdq)
            arima_model_fit = arima_model.fit()
            print(f"Критерий Акаике для pdq={pdq}: {arima_model_fit.aic}")
        
            # predict не работает с временными индексами - дает ошибку, поэтому start, end - порядковые
            arima_predict = arima_model_fit.predict(
                start=df_train.shape[0], #df_test.index[0], 
                end=df_train.shape[0] + df_test.shape[0] - 1#end=df_test.index[-1]
            )
        
            # Возвращение к индексам-датам
            arima_predict = pd.DataFrame({'model_preds': arima_predict})
            arima_predict['Date'] = df_test.index
        
            # DF с доверительным интервалом
            forecast = arima_model_fit.get_forecast(len(df_test.index))
            forecast_df = forecast.conf_int(alpha=0.05)
        
            # Объединение DF
            arima_predict = pd.concat([arima_predict, forecast_df], axis=1).set_index('Date')
        
        return arima_predict
    
    @staticmethod
    def sarima_predict_df(df_train, df_test, pdq=(1, 1, 1), PDQs=(1, 1, 1, 5)):
        """Формирует DF c предсказанием по SARIMA и доверительным интервалом"""
        # Отключение предупреждений
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")
            
            sarima_model = SARIMAX(df_train, order=pdq, seasonal_order=PDQs)
            sarima_model_fit = sarima_model.fit()
            print(f"Критерий Акаике для pdq={pdq}, PDQs={PDQs}: {sarima_model_fit.aic}")
        
            # predict не работает с временными индексами - дает ошибку, поэтому start, end - порядковые
            sarima_predict = sarima_model_fit.predict(
                start=df_train.shape[0], #df_test.index[0], 
                end=df_train.shape[0] + df_test.shape[0] - 1#end=df_test.index[-1]
            )
        
            # Возвращение к индексам-датам
            sarima_predict = pd.DataFrame({'model_preds': sarima_predict})
            sarima_predict['Date'] = df_test.index
        
            # DF с доверительным интервалом
            forecast = sarima_model_fit.get_forecast(len(df_test.index))
            forecast_df = forecast.conf_int(alpha=0.05)
        
            # Объединение DF
            sarima_predict = pd.concat([sarima_predict, forecast_df], axis=1).set_index('Date')
        
        return sarima_predict
    
    @staticmethod
    def plt_prediction(df, file_name=''):
        """Строит график реального DF + предсказанный фрагмент с доверительным интервалом"""
        # plt.rcParams["figure.figsize"] 18, 12

        # fig = plt.figure()
    
        # plt.subplot(2, 1, 1)
        # plt.plot(df["Close"], color = "blue", label = "Actuals", alpha = 0.4)
        # plt.plot(df.loc[test_df.index]["model_preds"], color = "red", linestyle = "-", label = "Out of Sample Fit")
        # plt.plot(df.loc[test_df.index]["model_preds_lower"], color = "green", linestyle = "--", label = "Confidence Intervals (95%)", alpha = 0.4)
        # plt.plot(df.loc[test_df.index]["model_preds_upper"], color = "green", linestyle = "--", alpha = 0.4)
        # plt.title("Full Model Fit", size = 24)
        # plt.grid()
        # plt.legend()
        
        # plt.subplot(2, 2, 3)
        # plt.plot(df.loc[test_df.index]["Close"], color = "blue", label = "Actuals", alpha = 0.6)
        # plt.plot(df.loc[test_df.index]["model_preds"], color = "red", linestyle = "-", label = "Out of Sample Fit", alpha = 0.6)
        # plt.plot(df.loc[test_df.index]["model_preds_lower"], color = "green", linestyle = "--", label = "Confidence Intervals (95%)", alpha = 0.6)
        # plt.plot(df.loc[test_df.index]["model_preds_upper"], color = "green", linestyle = "--", alpha = 0.6)
        # plt.title("Out of Sample Fit", size = 24)
        # plt.grid()
        # plt.legend()
        
        # preds_len = df['model_preds'].isna().sum()
        
        # plt.subplot(2, 2, 4)
        # plt.plot(df.iloc[preds_len-5:preds_len+30]["Close"], color = "blue", label = "Actuals", alpha = 0.6)
        # plt.plot(df.iloc[preds_len-5:preds_len+30]["model_preds"], color = "red", linestyle = "-", label = "Out of Sample Fit", alpha = 0.6)
        # plt.plot(df.iloc[preds_len-5:preds_len+30]["model_preds_lower"], color = "green", linestyle = "--", label = "Confidence Intervals (95%)", alpha = 0.6)
        # plt.plot(df.iloc[preds_len-5:preds_len+30]["model_preds_upper"], color = "green", linestyle = "--", alpha = 0.6)
        # plt.title("Out of Sample Fit", size = 24)
        # plt.grid()
        # plt.legend()
        
        # plt.show()
        pass

    @staticmethod
    def garch_prediction():
        """Предсказание волатильности ряда"""
        pass


    def make_prediction(self):
        """Предсказание методом ARIMA"""
        # self.train_test_groups()
        print('make_prediction')
    

class ModelSARIMAX(StockPrices):
    def make_prediction(self):
        """Предсказание методом SARIMAX"""
        pass


class ModelRegression(StockPrices):
    def make_prediction(self):
        """Предсказание """
        pass


class ModelNN(StockPrices):
    pass


# class ModelComparison(StockPrices, ModelARIMA, ModelSARIMAX, ModelRegression, ModelNN):
#     pass

