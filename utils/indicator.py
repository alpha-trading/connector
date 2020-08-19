# -*- coding:utf-8 -*-
from pandas import DataFrame
import numpy as np


class Indicator:
    @staticmethod
    def add_sma(df: DataFrame, *days) -> DataFrame:
        """
        단순 이동평균선(평균값으로) 구하기
        Example: add_sma(df, 5, 10)
        :param df:
        :param days: 날짜별 이동평균선 구하기
        :return:
        """
        for day in days:
            df[f'sma_{day}'] = df['close'].rolling(window=day).mean()
        return df

    @staticmethod
    def add_ema(df: DataFrame, *days):
        """
        지수 이동평균선(최근일에 가중치를 주는 방식) 구하기
        Example: add_ema(df, 5, 10)
        :param df:
        :param days:
        :return:
        """
        for day in days:
            df[f'ema_{day}'] = df['close'].ewm(span=day, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: DataFrame, *days) -> DataFrame:
        """
        RSI 과매도 과매수 판단 지표
        Example add_rsi(df, 14)
        :param df:
        :param days:
        :return:
        """
        for day in days:
            up = np.where(df['close'].diff(1) > 0, df['close'].diff(1), 0)
            down = np.where(df['close'].diff(1) < 0, df['close'].diff(1) * (-1), 0)

            average_up = DataFrame(up).rolling(window=day, min_periods=day).mean()
            average_down = DataFrame(down).rolling(window=day, min_periods=day).mean()
            df.reset_index(drop=False, inplace=True)
            df[f'rsi_{day}'] = average_up.div(average_down + average_up)
            df = df.set_index('date')
            try:
                df = df.drop(columns=['index'])
            except KeyError:
                pass
        return df

    @staticmethod
    def add_ibs(df: DataFrame):
        """
        IBS 구하기
        Example add_ibs(df)
        :param df:
        :return:
        """
        df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        return df

    @staticmethod
    def add_stddev(df: DataFrame, *days):
        """
        종가 기준으로 표준편차 구하기
        Example add_stddev(df, 5, 10, 20)
        :param df:
        :param days:
        :return:
        """
        for day in days:
            df[f'stddev_{day}'] = df['close'].rolling(window=day).std()
        return df

    @staticmethod
    def add_bollinger(df: DataFrame, *days):
        """
        볼린저 밴드
        Example: Indicator.add_bollinger(df, (20, 2), (60,2))
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
        sma + upper and lower bounds
        Example: add_envelope(df, (20, 0.05), (50, 0.05))
        """
        for day, r in days:
            line_mid = df['close'].rolling(window=day).mean()
            df[f'envelope_{day}_upper'] = line_mid + r * line_mid
            df[f'envelope_{day}_lower'] = line_mid - r * line_mid

        return df

    @staticmethod
    def add_dmi_atr(df: DataFrame, *days):
        """
        dmi: 추세판단
        atr: 위험판단
        Example: add_dmi_atr(df, 5)
        :param df:
        :param days:
        :return:
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

    @staticmethod
    def add_pivot(df: DataFrame):
        """
        전일 가격으로 pivot 중심선을 구하고 이를 통해 저항선 및 지지선 계산
        back test 할 때 다른 지표와는 달리 전일이 아닌 오늘 날짜를 사용해야 함.
        Example: add_pivot(df)
        :param df:
        :return:
        """
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r2'] = df['pivot'] + df['high'].shift(1) - df['low'].shift(1)
        df['r1'] = df['pivot'] * 2 - df['low'].shift(1)
        df['s1'] = df['pivot'] * 2 - df['high'].shift(1)
        df['s2'] = df['pivot'] - df['high'].shift(1) + df['low'].shift(1)

        return df

    @staticmethod
    def add_volume_ratio(df: DataFrame, *days):
        """
        하락한 날의 거래량 대비 상승한 날의 거래량 측정
        Example : add_volume_ratio(df, 20)
        :param df:
        :param days:
        :return:
        """
        for day in days:
            up = np.where(df['close'].diff(1) > 0, df['vol'], 0)
            down = np.where(df['close'].diff(1) < 0, df['vol'], 0)
            maintain = np.where(df['close'].diff(1) == 0, df['vol'] * 0.5, 0)
            up = up + maintain
            down = down + maintain
            sum_up = DataFrame(up).rolling(window=day, min_periods=day).sum()
            sum_down = DataFrame(down).rolling(window=day, min_periods=day).sum()
            df.reset_index(drop=False, inplace=True)
            df[f'vr_{day}'] = sum_up.div(sum_down)
            df = df.set_index('date')
            try:
                df = df.drop(columns=['index'])
            except KeyError:
                pass

        return df

    @staticmethod
    def add_atr(df: DataFrame, *days):
        """
        변동성 지표
        tr = max(고가 - 저가, abs(고가 - 전일 종가), (저가 - 전일 종가))
        tr을 평균을 내준다.
        Example:
        :param df:
        :param days:
        :return:
        """
        temp = df.copy()
        temp['a1'] = df['high'] - df['low']
        temp['a2'] = abs(df['close'].shift(1) - df['high'])
        temp['a3'] = abs(df['close'].shift(1) - df['low'])
        temp['tr'] = temp[['a1', 'a2', 'a3']].max(axis=1)

        for day in days:
            df[f'atr_{day}'] = temp['tr'].rolling(window=day, min_periods=day).mean()

        return df

    @staticmethod
    def add_range(df: DataFrame, *days):
        """
        고가 - 저가
        Example: add_range(df), add_range(df, 5)
        :param df:
        :param days:
        :return:
        """
        if not days:
            days = (1,)
        for day in days:
            df[f'range_{day}'] = (df['high'] - df['low']).rolling(window=day).mean()
        return df
