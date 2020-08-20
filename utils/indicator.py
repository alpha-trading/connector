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
    def get_ema(df: DataFrame, day: int, column='close') -> Series:
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
    def get_stddev(df: DataFrame, day: int, ddof: int = 1) -> Series:
        """
        종가 기준으로 표준편차 구하기
        Example add_stddev(df, 5)
        :param df:
        :param day:
        :param ddof:
        :return:
        """
        return df['close'].rolling(window=day).std(ddof=ddof)

    @classmethod
    def get_bollinger(cls, df: DataFrame, day: int, r: int) -> Tuple[Series, Series]:
        """
        볼린저 밴드
        Example: Indicator.add_bollinger(df, 20, 2)
        """
        line_mid = cls.get_sma(df, day)
        line_std = cls.get_stddev(df, day, ddof=0)

        upper = line_mid + r * line_std
        lower = line_mid - r * line_std

        return upper, lower

    @classmethod
    def get_envelope(cls, df: DataFrame, day: int, r: float) -> Tuple[Series, Series]:
        """
        sma + upper and lower bounds
        Example: add_envelope(df, 20, 0.05)
        """
        line_mid = cls.get_sma(df, day)
        upper = line_mid + r * line_mid
        lower = line_mid - r * line_mid

        return upper, lower

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
        sum_up = Series(up).rolling(window=day, min_periods=day).sum()
        sum_down = Series(down).rolling(window=day, min_periods=day).sum()
        return sum_up.div(sum_down)

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

    @staticmethod
    def get_demark(df: DataFrame) -> Series:
        """
        :param df:
        :return: demark 고가, demark 저가
        """
        d1 = np.where(df['close'] > df['open'], (df['high'] * 2 + df['low'] + df['close']) / 2, 0)
        d2 = np.where(df['close'] < df['open'], (df['high'] + df['low'] * 2 + df['close']) / 2, 0)
        d3 = np.where(df['close'] == df['open'], (df['high'] + df['low'] + df['close'] * 2) / 2, 0)
        d = Series(d1 + d2 + d3)
        demark_high = (d - df['low']).shift(1)
        demark_low = (d - df['high']).shift(1)

        return demark_high, demark_low
