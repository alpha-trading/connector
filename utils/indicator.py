# -*- coding:utf-8 -*-
from typing import Tuple

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from enum import Enum


class Criteria(Enum):
    sma = 1
    ema = 2
    ewma = 3
    wma = 4


class Indicator:

    @staticmethod
    def get_sma(price: Series, day: int) -> Series:
        return price.rolling(window=day).mean()

    @staticmethod
    def get_ema(price: Series, day: int) -> Series:
        ema = price.ewm(span=day, min_periods=day, adjust=False).mean()
        return ema

    @staticmethod
    def get_ewma(price: Series, day: int) -> Series:
        ewma = price.ewm(span=day, min_periods=day, adjust=True).mean()
        return ewma

    @staticmethod
    def get_wma(price: Series, day: int) -> Series:
        weight = np.arange(1, day+1)
        wma = price.rolling(window=day).apply(lambda prices: np.dot(prices, weight) / weight.sum(), raw=True)
        return wma

    @staticmethod
    def get_rsi(price: Series, day: int) -> Series:
        up = np.where(price.diff(1).gt(0), price.diff(1), 0)
        down = np.where(price.diff(1).lt(0), price.diff(1).multiply(-1), 0)
        average_up = Series(up).rolling(window=day, min_periods=day).mean()
        average_down = Series(down).rolling(window=day, min_periods=day).mean()
        return average_up.div(average_down + average_up)

    @staticmethod
    def get_ibs(high: Series, low: Series, close: Series) -> Series:
        return (close - low) / (high - low)

    @staticmethod
    def get_stddev(price: Series, day: int, ddof: int = 1) -> Series:
        return price.rolling(window=day).std(ddof=ddof)

    @classmethod
    def get_bollinger(cls, price: Series, day: int, r: int, criteria: Criteria = Criteria.ema) -> Tuple[Series, Series]:
        if criteria == Criteria.sma:
            line_mid = cls.get_sma(price, day)
        elif criteria == Criteria.ema:
            line_mid = cls.get_ema(price, day)
        elif criteria == Criteria.ewma:
            line_mid = cls.get_ewma(price, day)
        elif criteria == Criteria.wma:
            line_mid = cls.get_wma(price, day)
        line_std = cls.get_stddev(price, day, ddof=0)
        bollinger_range = line_std.multiply(r)
        upper = line_mid + bollinger_range
        lower = line_mid - bollinger_range
        return upper, lower

    @classmethod
    def get_envelope(cls, price: Series, day: int, r: float, criteria: Criteria = Criteria.ema) -> Tuple[Series, Series]:
        if criteria == Criteria.sma:
            line_mid = Indicator.get_sma(price, day)
        elif criteria == Criteria.ema:
            line_mid = Indicator.get_ema(price, day)
        elif criteria == Criteria.ewma:
            line_mid = Indicator.get_ewma(price, day)
        elif criteria == Criteria.wma:
            line_mid = Indicator.get_wma(price, day)

        envelope_range = line_mid.multiply(r)
        upper = line_mid + envelope_range
        lower = line_mid - envelope_range
        return upper, lower

    @staticmethod
    def get_pivot(high: Series, low: Series, close: Series) -> Tuple[Series, Series, Series, Series, Series]:
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        r2 = pivot + high.shift(1).sub(low.shift(1))
        r1 = (pivot * 2).sub(low.shift(1))
        s1 = (pivot * 2).sub(high.shift(1))
        s2 = pivot.sub(high.shift(1)) + low.shift(1)

        return r2, r1, pivot, s1, s2

    @staticmethod
    def get_volume_ratio(close: Series, volume: Series, day: int) -> Series:
        up = np.where(close.diff(1).gt(0), volume, 0)
        down = np.where(close.diff(1).lt(0), volume, 0)
        maintain = np.where(close.diff(1).equals(0), volume.mul(0.5), 0)

        up = up + maintain
        down = down + maintain
        sum_up = Series(up).rolling(window=day, min_periods=day).sum()
        sum_down = Series(down).rolling(window=day, min_periods=day).sum()
        return sum_up.div(sum_down)

    @classmethod
    def get_range(cls, high: Series, low: Series, day: int, criteria: Criteria = Criteria.sma) -> Series:
        if criteria == Criteria.sma:
            day_range = cls.get_sma(high - low, day)
        elif criteria == Criteria.ema:
            day_range = cls.get_ema(high - low, day)
        elif criteria == Criteria.ewma:
            day_range = cls.get_ewma(high - low, day)
        elif criteria == Criteria.wma:
            day_range = cls.get_wma(high - low, day)
        return day_range

    @staticmethod
    def get_demark(open: Series, high: Series, low: Series, close: Series) -> Tuple[Series, Series]:
        d1 = np.where(close > open, (high.mul(2) + low + close) / 2, 0)
        d2 = np.where(close < open, (high + low.mul(2) + close) / 2, 0)
        d3 = np.where(close == open, (high + low + close.mul(2)) / 2, 0)
        d = Series(d1 + d2 + d3)
        demark_high = (d - low).shift(1)
        demark_low = (d - high).shift(1)

        return demark_high, demark_low

    @staticmethod
    def get_momentum(price: Series, day: int) -> Series:
        return price.diff(day) / price.shift(day)

    @staticmethod
    def get_psychological_line(price: Series, day: int) -> Series:
        up = np.where(price.diff(1).gt(0), 1, 0)
        sum_up = Series(up).rolling(window=day, min_periods=day).sum()

        psychological = sum_up.divide(day)
        return psychological

    @staticmethod
    def get_new_psychological_line(price: Series, day: int) -> Series:
        up_cnt = np.where(price.diff(1).gt(0), 1, 0)
        up_cnt_cum = Series(up_cnt).rolling(window=day, min_periods=day).sum()
        up_width = np.where(price.diff(1).gt(0), price.diff(1), 0)
        up_width_cum = Series(up_width).rolling(window=day, min_periods=day).sum()

        down_cnt = np.where(price.diff(1).lt(0), 1, 0)
        down_cnt_cum = Series(down_cnt).rolling(window=day, min_periods=day).sum()
        down_width = np.where(price.diff(1).lt(1), abs(price.diff(1)), 0)
        down_width_cum = Series(down_width).rolling(window=day, min_periods=day).sum()

        up = up_cnt_cum.multiply(up_width_cum)
        down = down_cnt_cum.multiply(down_width_cum)

        quo = up.subtract(down)
        deno = (up_width_cum + down_width_cum).multiply(day)

        psychological = quo.divide(deno)
        return psychological

    @classmethod
    def get_disparity(cls, price: Series, day: int, criteria: Criteria = Criteria.sma) -> Series:
        if criteria == Criteria.sma:
            moving_average = cls.get_sma(price, day)
        elif criteria == Criteria.ema:
            moving_average = cls.get_ema(price, day)
        elif criteria == Criteria.wma:
            moving_average = cls.get_wma(price, day)
        disparity = price.divide(moving_average)
        return disparity

    @classmethod
    def get_dmi(cls, high: Series, low: Series, close: Series, day: int) -> Tuple[Series, Series]:
        data = {'range': abs(high - low), 'up': abs(high - close.shift(1)), 'down': abs(close.shift(1) - low)}
        data = DataFrame(data, columns=['range', 'up', 'down'])

        tr = data.max(axis=1)

        pdm = np.where(((high.diff(1) > 0) & (high.diff(1) > low.shift(1) - low)), high.diff(1), 0)
        pdm = cls.get_ema(Series(pdm), day)
        mdm = np.where((((low.shift(1) - low) > 0) & (high.diff(1) < (low.shift(1) - low))), low.shift(1) - low, 0)
        mdm = cls.get_ema(Series(mdm), day)

        div = cls.get_ema(tr, day)

        pdi = pdm.divide(div)
        mdi = mdm.divide(div)
        dx = abs(pdi - mdi).divide(pdi + mdi)
        adx = cls.get_ema(dx, day)
        return pdi, mdi, adx

    @staticmethod
    def get_price_channel(high: Series, low: Series, day: int) -> Tuple[Series, Series]:

        resistance_line = high.shift(1).rolling(window=day).max()
        support_line = low.shift(1).rolling(window=day).min()
        return resistance_line, support_line

    @classmethod
    def get_macd(cls, price: Series, day_short: int, day_long: int, criteria: Criteria = Criteria.ewma) -> Series:
        if criteria == Criteria.sma:
            short_term = cls.get_sma(price, day_short)
            long_term = cls.get_sma(price, day_long)
        elif criteria == Criteria.ema:
            short_term = cls.get_ema(price, day_short)
            long_term = cls.get_ema(price, day_long)
        elif criteria == Criteria.ewma:
            short_term = cls.get_ewma(price, day_short)
            long_term = cls.get_ewma(price, day_long)
        elif criteria == Criteria.wma:
            short_term = cls.get_wma(price, day_short)
            long_term = cls.get_wma(price, day_long)

        macd = short_term - long_term
        return macd

    @classmethod
    def get_macd_osillator(cls, price: Series, day_short: int, day_long: int, day_signal: int, criteria: Criteria = Criteria.ewma) -> Series:
        macd = cls.get_macd(price, day_short, day_long, criteria)
        if criteria == Criteria.sma:
            signal = cls.get_sma(macd, day_signal)
        elif criteria == Criteria.ema:
            signal = cls.get_ema(macd, day_signal)
        elif criteria == Criteria.ewma:
            signal = cls.get_ewma(macd, day_signal)
        elif criteria == Criteria.wma:
            signal = cls.get_wma(macd, day_signal)

        macd_osillator = macd - signal
        return macd_osillator
