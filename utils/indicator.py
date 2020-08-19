# -*- coding:utf-8 -*-
from typing import Tuple

from pandas import DataFrame, Series
import numpy as np


class Indicator:
    @staticmethod
    def get_sma(df: DataFrame, day: int, column='close') -> Series:
        """
        단순 이동평균선(평균값으로) 구하기
        Example: add_sma(df, 10)
        :param column:
        :param df:
        :param day: 일자별 이동평균선 구하기
        :return:
        """
        return df[column].rolling(window=day).mean()

    @staticmethod
    def get_ema(df: DataFrame, day: int, column='close') ->Series:
        """
        지수 이동평균선(최근일에 가중치를 주는 방식) 구하기
        Example: add_ema(df, 5)
        :param column:
        :param df:
        :param day:
        :return:
        """
        return df[column].ewm(span=day, adjust=False).mean()

    @staticmethod
    def get_rsi(df: DataFrame, day: int) -> Series:
        """
        RSI 과매도 과매수 판단 지표
        Example add_rsi(df, 14)
        :param df:
        :param day:
        :return:
        """
        up = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
        down = np.where(df['close'].diff(1) < 0, df['close'].diff(1) * (-1), 0)

        average_up = Series(up).rolling(window=day, min_periods=day).mean()
        average_down = Series(down).rolling(window=day, min_periods=day).mean()
        return average_up.div(average_down + average_up)

    @staticmethod
    def get_ibs(df: DataFrame) -> Series:
        """
        IBS 구하기
        Example add_ibs(df)
        :param df:
        :return:
        """
        return (df['close'] - df['low']) / (df['high'] - df['low'])

    @staticmethod
    def get_stddev(df: DataFrame, day: int) -> Series:
        """
        종가 기준으로 표준편차 구하기
        Example add_stddev(df, 5)
        :param df:
        :param day:
        :return:
        """
        return  df['close'].rolling(window=day).std()

    @staticmethod
    def get_bollinger(df: DataFrame, day: int, r: int) -> Tuple[Series, Series, Series]:
        """
        볼린저 밴드
        Example: Indicator.add_bollinger(df, 20, 2)
        """
        line_mid = df['close'].rolling(window=day).mean()
        line_std = df['close'].rolling(window=day).std()

        upper = line_mid + r * line_std
        lower = line_mid - r * line_std
        width = (df[f'bollin_{day}_upper'] - df[f'bollin_{day}_lower']) / line_mid

        return upper, lower, width

    @staticmethod
    def get_envelope(df: DataFrame, day: int, r: float) -> Tuple[Series, Series]:
        """
        sma + upper and lower bounds
        Example: add_envelope(df, 20, 0.05)
        """
        line_mid = df['close'].rolling(window=day).mean()
        upper = line_mid + r * line_mid
        lower = line_mid - r * line_mid

        return upper, lower

    @classmethod
    def get_dmi(cls, df: DataFrame, day: int) -> Tuple[Series, Series, Series]:
        """
        dmi: 추세판단
        atr: 위험판단
        Example: add_dmi_atr(df, 5)
        :param df:
        :param day:
        :return: plus di, minus di, adx
        """
        temp = df.copy()
        temp['pdm'] = np.where((df['high'].diff() > 0) & (df['high'].diff() + df['low'].diff() > 0),
                               df['high'].diff(), 0)
        temp['mdm'] = np.where((df['low'].diff() < 0) & (df['high'].diff() + df['low'].diff() < 0),
                               abs(df['low'].diff()), 0)

        temp['atr'] = cls.get_atr(temp, day)
        temp['pdm_ema'] = cls.get_ema(temp, day, column='pdm')
        temp['mdm_ema'] = cls.get_ema(temp, day, column='mdm')

        pdi = temp['pdm_ema'] / temp['atr']
        mdi = temp['mdm_ema'] / temp['atr']
        temp['dx'] = abs(pdi - mdi) / (pdi + mdi)
        adx = temp['dx'].rolling(window=day).mean()

        return pdi, mdi, adx

    @staticmethod
    def get_pivot(df: DataFrame) -> Tuple[Series, Series, Series, Series, Series]:
        """
        전일 가격으로 pivot 중심선을 구하고 이를 통해 저항선 및 지지선 계산
        back test 할 때 다른 지표와는 달리 전일이 아닌 오늘 날짜를 사용해야 함.
        Example: add_pivot(df)
        :param df:
        :return: 2차 저항선, 1차 저항선, 피봇 중심선, 1차 지지선, 2차 지지
        """
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        r2 = pivot + df['high'].shift(1) - df['low'].shift(1)
        r1 = pivot * 2 - df['low'].shift(1)
        s1 = pivot * 2 - df['high'].shift(1)
        s2 = pivot - df['high'].shift(1) + df['low'].shift(1)

        return r2, r1, pivot, s1, s2

    @staticmethod
    def get_volume_ratio(df: DataFrame, day: int) -> Series:
        """
        하락한 날의 거래량 대비 상승한 날의 거래량 측정
        Example : add_volume_ratio(df, 20)
        :param df:
        :param day:
        :return:
        """
        up = np.where(df['close'].diff(1) > 0, df['vol'], 0)
        down = np.where(df['close'].diff(1) < 0, df['vol'], 0)
        maintain = np.where(df['close'].diff(1) == 0, df['vol'] * 0.5, 0)
        up = up + maintain
        down = down + maintain
        sum_up = DataFrame(up).rolling(window=day, min_periods=day).sum()
        sum_down = DataFrame(down).rolling(window=day, min_periods=day).sum()
        return sum_up.div(sum_down)

    @staticmethod
    def wwma(values:Series, day: int) -> Series:
        """
         J. Welles Wilder's EMA
        """
        return values.ewm(alpha=1 / day, min_periods=day, adjust=False).mean()

    @classmethod
    def get_atr(cls, df: DataFrame, day: int) -> Series:
        """
        변동성 지표
        tr = max(고가 - 저가, abs(고가 - 전일 종가), (저가 - 전일 종가))
        tr을 평균을 내준다.
        Example:
        :param df:
        :param day:
        :return:
        """
        temp = df.copy()
        temp['a1'] = df['high'] - df['low']
        temp['a2'] = abs(df['close'].shift(1) - df['high'])
        temp['a3'] = abs(df['close'].shift(1) - df['low'])
        temp['tr'] = temp[['a1', 'a2', 'a3']].max(axis=1)
        return cls.wwma(temp['tr'], day)

    @classmethod
    def get_eatr(cls, df: DataFrame, day: int):
        """
        변동성 지표
        tr = max(고가 - 저가, abs(고가 - 전일 종가), (저가 - 전일 종가))
        tr을 지수 가중 평균을 내준다.
        Example:
        :param df:
        :param day:
        :return:
        """
        temp = df.copy()
        temp['a1'] = df['high'] - df['low']
        temp['a2'] = abs(df['close'].shift(1) - df['high'])
        temp['a3'] = abs(df['close'].shift(1) - df['low'])
        temp['tr'] = temp[['a1', 'a2', 'a3']].max(axis=1)

        return cls.get_ema(temp, day)

    @staticmethod
    def get_range(df: DataFrame, day: int = 1):
        """
        고가 - 저가
        Example: add_range(df), add_range(df, 5)
        :param df:
        :param day:
        :return:
        """
        return (df['high'] - df['low']).rolling(window=day).mean()
