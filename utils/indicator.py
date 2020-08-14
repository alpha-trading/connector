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
        with_profit['profit_rate'] = pnl.delta / pnl.shift(1).pnl
        return with_profit

    @staticmethod
    def add_dma(df: DataFrame, *days) -> DataFrame:
        """이동평균선 구하기"""
        for day in days:
            df[f'dma_{day}'] = df['close'].rolling(window=day).mean()
        return df
    
    @staticmethod
    def add_ema(df: DataFrame, *days):
        for day in days:
            df[f'ema_{day}'] = df['close'].ewm(span=day, adjust=False).mean()
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
        사용법: Indicator.add_bollinger(df, (20, 2), (60,2))
        """
        for day, r in days:
            line_mid = df['close'].rolling(window=day).mean()
            line_std = df['close'].rolling(window=day).std()
            df[f'bollin_{day}_upper'] = line_mid + r * line_std
            df[f'bollin_{day}_lower'] = line_mid - r * line_std
            df[f'bollin_{day}_width'] = (df[f'bollin_{day}_upper'] - df[f'bollin_{day}_lower']) / line_mid

        return df
    
    @staticmethod
    def add_envelope(df: DataFrame, *days):
        """
        dma + upper and lower bounds
        사용법: Indicator.add_envelope(df, (20, 20), (50, 5))
        """
        for day, r in days:
            line_mid = df['close'].rolling(window=day).mean() 
            df[f'envelope_{day}_upper'] = line_mid + r / 100 * line_mid
            df[f'envelope_{day}_lower'] = line_mid - r / 100 * line_mid
            df[f'envelope_{day}_width'] = (df[f'envelope_{day}_upper'] - df[f'envelope_{day}_lower']) / line_mid
        
        return df

    @staticmethod
    def add_dmi_atr(df: DataFrame, *days):
        """
        dmi : 추세판단
        atr : 위험판단
        """
        for day in days:
            with_dmi = df.copy()
            with_dmi['dm_p'] = np.where(df['high'].diff(1) > 0 and df['high'].diff(1) + df['low'].diff(1) > 0,
                                        df['high'].diff(1), 0)
            with_dmi['dm_m'] = np.where(df['low'].diff(1) < 0 and df['high'].diff(1) + df['low'].diff(1) < 0,
                                        df['low'].diff(1) * (-1), 0)
            with_dmi['tr'] = df.max(df['high'] - df['low'], abs(df['close'].shift(-1) - df['high']),
                                    abs(df['close'].shift(-1) - df['low']))
            with_dmi['di_p'] = with_dmi['dm_p'].rolling(window=day).mean() / with_dmi['tr'].rolling(window=day).mean()
            with_dmi['di_m'] = with_dmi['dm_m'].rolling(window=day).mean() / with_dmi['tr'].rolling(window=day).mean()
            df[f'atr_{day}'] = with_dmi['tr'].rolling(window=day).mean()
            df[f'dx_{day}'] = abs(with_dmi['di_p'] - with_dmi['di_m']) / (with_dmi['di_p'] + with_dmi['di_m'])
            df[f'adx_{day}'] = df[f'dx_{day}'].rolling(window=day).mean()
            df[f'adxr_{day}'] = df[f'adx_{day}'].rolling(window=day).mean()

        return df
