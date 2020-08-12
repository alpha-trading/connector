# -*- coding:utf-8 -*-
from pandas import DataFrame
import numpy as np


class Indicator:
    @staticmethod
    def add_delta(pnl: DataFrame) -> DataFrame:
        """종가 변화 컬럼 추가"""
        with_delta = pnl.copy()
        with_delta['delta'] = pnl['close'].diff().shift(1)
        return with_delta

    @staticmethod
    def add_profit_rate(pnl: DataFrame) -> DataFrame:
        """이익율 컬럼 추가"""
        with_profit = pnl.copy()
        with_profit['profit_rate'] = pnl.delta / pnl.shift(1).pnl * 100
        return with_profit

    @staticmethod
    def add_dma(df: DataFrame, *days) -> DataFrame:
        """이동평균선 구하기"""
        for day in days:
            df[f'dma_{day}'] = df['close'].rolling(window=day).mean()
        return df

    @staticmethod
    def add_rsi(df: DataFrame, *days) -> DataFrame:
        for day in days:
            up = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
            down = np.where(df['close'].diff(1) < 0, df['close'].diff(1) * (-1), 0)

            average_up = DataFrame(up).rolling(window=day, min_periods=day).mean()
            average_down = DataFrame(down).rolling(window=day, min_periods=day).mean()
            df[f'rsi_{day}'] = average_up.div(average_down + average_up)

        return df

    @staticmethod
    def add_ibs(df: DataFrame):
        df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        return df

    @staticmethod
    def add_stddev(df: DataFrame, *days):
        for day in days:
            df[f'stddev_{day}'] = df['close'].rolling(window=day).std()
        return df

    @staticmethod
    def add_range(df: DataFrame):
        df['range'] = df['high'] - df['low']
        return df

    @staticmethod
    def add_bollinger(df: DataFrame, *days):
        """
        볼린저 밴드
        사용법: Indicator.add_bollinger(df, (20, 2), (60,2)
        """
        for day, r in days:
            line_mid = df['close'].rolling(window=day).mean()
            line_std = df['close'].rolling(window=day).std()
            df[f'bollin_{day}_upper'] = line_mid + line_std
            df[f'bollin_{day}_lower'] = line_mid - line_std
            df[f'bollin_{day}_width'] = (df[f'bollin_{day}_upper'] - df[f'bollin_{day}_lower']) / line_mid

        return df
