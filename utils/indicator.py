# -*- coding:utf-8 -*-
from typing import Tuple
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
        """
        단순이동평균을 구하는 함수
        :param price: 단순이동평균을 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일간 price의 단순이동평균
        """
        return price.rolling(window=day).mean()

    @staticmethod
    def get_ema(price: Series, day: int) -> Series:
        """
        지수이동평균을 구하는 함수
        :param price: 지수이동평균을 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일간 price의 지수이동평균
        """
        ema = price.ewm(span=day, min_periods=day, adjust=False).mean()
        return ema

    @staticmethod
    def get_ewma(price: Series, day: int) -> Series:
        ewma = price.ewm(span=day, min_periods=day, adjust=True).mean()
        return ewma

    @staticmethod
    def get_wma(price: Series, day: int) -> Series:
        """
        가중이동평균을 구하는 함수
        :param price: 가중이동평균을 구할 떄 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일간 price의 가중이동평균
        """
        weight = np.arange(1, day+1)
        wma = price.rolling(window=day).apply(lambda prices: np.dot(prices, weight) / weight.sum(), raw=True)
        return wma

    @staticmethod
    def get_stddev(price: Series, day: int, ddof: int = 1) -> Series:
        return price.rolling(window=day).std(ddof=ddof)

    @classmethod
    def get_bollinger(cls, price: Series, day: int, r: int, criteria: Criteria = Criteria.ewma) -> Tuple[Series, Series]:
        """
        Bollinger Bands를 구하는 함수
        Bollinger Bands의 상, 하한선은 표준 편차에 의해 산출된 이동평균 값이며,
        주가나 지수의 움직임이 큰 시기에는 Bands의 폭이 넓어지고 움직임이 작은 시기에는 Bands의 폭이 좁아지는 특정을 가지고 있다.
        따라서, 가격 움직임의 크기에 따라 밴드의 넓이가 결정된다.
        :param price: Bollinger Bands를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :param r: 표준편차 승수 ex) 2
        :param criteria: 중간 밴드 기준 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: Bollinger Bands 상한선, 하한선
        """
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

    @staticmethod
    def get_demark(open: Series, high: Series, low: Series, close: Series) -> Tuple[Series, Series]:
        """
        Demark를 구하는 함수
        :param open: 시가
        :param high: 고가
        :param low: 저가
        :param close: 종가
        :return: Demark 저항선, 지지선
        """
        d1 = np.where(close > open, (high.mul(2) + low + close) / 2, 0)
        d2 = np.where(close < open, (high + low.mul(2) + close) / 2, 0)
        d3 = np.where(close == open, (high + low + close.mul(2)) / 2, 0)
        d = Series(d1 + d2 + d3)
        demark_high = (d - low).shift(1)
        demark_low = (d - high).shift(1)

        return demark_high, demark_low

    @classmethod
    def get_envelope(cls, price: Series, day: int, r: float, criteria: Criteria = Criteria.sma) -> Tuple[Series, Series]:
        """
        Envelope를 구하는 함수
        :param price: Envelope를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :param r: 비율 ex) 0.02, 0.08
        :param criteria: 중심선 기준 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: Envelope 상한선, 하한선
        """
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
        """
        Pivot을 구하는 함수
        :param high: 고가
        :param low: 저가
        :param close: 종가
        :return: 2차 저항선, 1차 저항선, Pivot 중심선, 1차 지지선, 2차 지지선
        """
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        r2 = pivot + high.shift(1).sub(low.shift(1))
        r1 = (pivot * 2).sub(low.shift(1))
        s1 = (pivot * 2).sub(high.shift(1))
        s2 = pivot.sub(high.shift(1)) + low.shift(1)

        return r2, r1, pivot, s1, s2

    @staticmethod
    def get_price_channel(high: Series, low: Series, day: int) -> Tuple[Series, Series]:
        """
        Price Channel를 구하는 함수
        Price Channel은 저항선과 지지선으로 구성된다.
        저항선은 일정기간전의 최고가를, 지지선은 일정기간전의 최저가를 이은선이다.
        이 채널은 채널포지션을 결정하는데 사용되지 않고 현재바의 최저가와 최고가를 보여줄 뿐이다.
        저항선은 뚜렷한 시장세력의 신호이며, 지지선은 뚜렷한 시장약화의 신호이다.
        :param high: 고가
        :param low: 저가
        :param day:
        :return: Price Channel 저항선, 지지선
        """
        resistance_line = high.shift(1).rolling(window=day).max()
        support_line = low.shift(1).rolling(window=day).min()
        return resistance_line, support_line

    @classmethod
    def get_dmi(cls, high: Series, low: Series, close: Series, day: int) -> Tuple[Series, Series]:
        """
        DMI를 구하는 함수
        DMI는 현재 시장의 추세와 강도를 함께 나타내는 지표로 단기보다는 중장기 추세분석에 유리하다.
        PDI는 실질적으로 상승하는 폭의 비율을 나타내며 MDI는 실질적으로 하락하는 폭의 비율을 의미한다.
        :param high: 고가
        :param low: 저가
        :param close: 종가
        :param day:
        :return: +DI, -DI, ADX
        """
        data = {'range': abs(high - low), 'up': abs(high - close.shift(1)), 'down': abs(close.shift(1) - low)}
        data = DataFrame(data, columns=['range', 'up', 'down'])

        tr = data.max(axis=1)

        pdm = np.where(((high.diff(1) > 0) & (high.diff(1) > low.shift(1) - low)), high.diff(1), 0)
        pdmn = cls.get_ema(Series(pdm), day)
        mdm = np.where((((low.shift(1) - low) > 0) & (high.diff(1) < (low.shift(1) - low))), low.shift(1) - low, 0)
        mdmn = cls.get_ema(Series(mdm), day)

        div = cls.get_ema(tr, day)

        pdi = pdmn.divide(div)
        mdi = mdmn.divide(div)
        dx = abs(pdi - mdi).divide(pdi + mdi)
        adx = cls.get_ema(dx, day)
        return pdi, mdi, adx

    @classmethod
    def get_macd(cls, price: Series, day_short: int, day_long: int, criteria: Criteria = Criteria.ewma) -> Series:
        """
        MACD(Moving Average Convergence and Divergence)를 구하는 함수
        :param price: MACD를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day_short: 단기이동평균 기간
        :param day_long: 장기이동평균 기간
        :param criteria: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: MACD
        """
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
    def get_macd_oscillator(cls, price: Series, day_short: int, day_long: int, day_signal: int,
                           criteria: Criteria = Criteria.ewma) -> Series:
        """
        MACD Oscillator를 구하는 함수
        MACD Oscillator는 MACD와 Signal의 교차를 보다 정확하게 인식하기 위해서 MACD 값에서 Signal 값을 뺀다.
        :param price: MACD Oscillator를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day_short: 단기이동평균 기간
        :param day_long: 장기이동평균 기간
        :param day_signal: 시그널 기간
        :param criteria: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: MACD Oscillator
        """
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

    @staticmethod
    def get_momentum(price: Series, day: int) -> Series:
        """
        Momentum를 구하는 함수
        :param price: Momentum를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일의 Momentum
        """
        return price.diff(day) / price.shift(day)

    @staticmethod
    def get_rsi(price: Series, day: int) -> Series:
        """
        RSI를 구하는 함수
        RSI는 시장가격의 변동폭 중에서 상승폭이 어느 정도인지를 분석하여 현재의 시장가격이 상승세라면 얼마나 강력한 상승 추세인지,
        그리고 하락세라면 얼마나 강력한 하락추세인지를 나타낸 것이다.
        추세의 강도를 표시함으로써 향후 추세전환시점의 예측을 가능하게 한다.
        :param price: RSI를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일간 RSI
        """
        up = np.where(price.diff(1).gt(0), price.diff(1), 0)
        down = np.where(price.diff(1).lt(0), price.diff(1).multiply(-1), 0)
        average_up = Series(up).rolling(window=day, min_periods=day).mean()
        average_down = Series(down).rolling(window=day, min_periods=day).mean()
        return average_up.div(average_down + average_up)

    @classmethod
    def get_stochastic_fast(cls, high: Series, low: Series, close: Series, fast_k_period: int, fast_d_period: int,
                            criteria: Criteria = Criteria.ema):
        percent_k = ((close - low.rolling(window=fast_k_period).min()) / (high.rolling(window=fast_k_period).max()
                                                                          - low.rolling(window=fast_k_period).min()))
        if criteria == Criteria.sma:
            percent_d = cls.get_sma(percent_k, fast_d_period)
        elif criteria == Criteria.ema:
            percent_d = cls.get_ema(percent_k, fast_d_period)
        elif criteria == Criteria.ewma:
            percent_d = cls.get_ewma(percent_k, fast_d_period)
        elif criteria == Criteria.wma:
            percent_d = cls.get_wma(percent_k, fast_d_period)

        return percent_k, percent_d

    @staticmethod
    def get_volume_ratio(close: Series, volume: Series, day: int) -> Series:
        """
        Volume Ratio를 구하는 함수
        Volume Ratio는 일정 기간 동안 시장가격 상승일의 거래량과 시장가격 하락일의 거래량을 비교하여 나타낸 지표로서,
        현재 시장이 과열인지 침체인지를 판단하게 해주는 시장특성 지표이다.
        :param close: 종가
        :param volume: 거래량
        :param day:
        :return: N일간 Volume Ratio
        """
        up = np.where(close.diff(1).gt(0), volume, 0)
        down = np.where(close.diff(1).lt(0), volume, 0)
        maintain = np.where(close.diff(1).equals(0), volume.mul(0.5), 0)

        up = up + maintain
        down = down + maintain
        sum_up = Series(up).rolling(window=day, min_periods=day).sum()
        sum_down = Series(down).rolling(window=day, min_periods=day).sum()
        return sum_up.div(sum_down)

    @staticmethod
    def get_psychological_line(price: Series, day: int) -> Series:
        """
        Psychological Line을 구하는 함수
        Psychological Line은 주식시장이 현재 과열 국면인지 침체 국면인지를 파악하여 단기적 매매시점을 결정하기 위한 지표로
        시장의 갑작스런 악재나 호재를 즉각 반영시킴으로써 시장의 변화를 신속하여 객관적으로 판단할 수 있는 지표이다.
        :param price: Psychological Line을 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일간 Psychological Line
        """
        up = np.where(price.diff(1).gt(0), 1, 0)
        sum_up = Series(up).rolling(window=day, min_periods=day).sum()

        psychological = sum_up.divide(day)
        return psychological

    @staticmethod
    def get_new_psychological_line(price: Series, day: int) -> Series:
        """
        New Psychological Line을 구하는 함수
        New Psychological Line은 과열 및 침체도를 파악하고자 하는 기법이다.
        Psychological Line은 주가가 상승한 날만을 가지고 판단하기 때문에 주가등락폭에 대한 시장의 심리는 반영하지 못하는 단점을 가지고 있으나
        New Psychological Line은 이같은 단점을 개선한 지표이다.
        :param price: New Psychological Line을 구할 떄 사용하는 가격 ex) open, high, low, close
        :param day:
        :return: N일간  New Psychological Line
        """
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
        """
        Disparity를 구하는 함수
        Disparity는 주가가 이동평균선과 어느 정도 떨어져 있는가를 분석한 것이다.
        :param price: Disparity를 구할 때 사용하는 가격 ex) open, high, low, close
        :param day:
        :param criteria: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: N일간 이동평균과의 Disparity
        """
        if criteria == Criteria.sma:
            moving_average = cls.get_sma(price, day)
        elif criteria == Criteria.ema:
            moving_average = cls.get_ema(price, day)
        elif criteria == Criteria.wma:
            moving_average = cls.get_wma(price, day)
        disparity = price.divide(moving_average)
        return disparity

    @staticmethod
    def get_ibs(high: Series, low: Series, close: Series) -> Series:
        """
        IBS를 구하는 함수
        IBS = (close - low) / (high - low)
        :param high: 고가
        :param low: 저가
        :param close: 종가
        :return: IBS
        """
        return (close - low) / (high - low)

    @classmethod
    def get_range(cls, high: Series, low: Series, day: int, criteria: Criteria = Criteria.sma) -> Series:
        """
        Range를 구하는 함수
        :param high: 고가
        :param low: 저가
        :param day:
        :param criteria: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: N일간 평균 Range
        """
        if criteria == Criteria.sma:
            day_range = cls.get_sma(high - low, day)
        elif criteria == Criteria.ema:
            day_range = cls.get_ema(high - low, day)
        elif criteria == Criteria.ewma:
            day_range = cls.get_ewma(high - low, day)
        elif criteria == Criteria.wma:
            day_range = cls.get_wma(high - low, day)
        return day_range

