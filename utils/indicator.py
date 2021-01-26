# -*- coding:utf-8 -*-
from typing import Tuple
from pandas import DataFrame, Series
import numpy as np

from utils.parameter import MovingAverage


class Indicator:
    @staticmethod
    def get_sma(price: Series, day: int) -> Series:
        """
        단순이동평균을 구하는 함수
        :param price: 단순이동평균을 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day:
        :return: N일간 price의 단순이동평균
        """
        return price.rolling(window=day).mean()

    @staticmethod
    def get_ema(price: Series, day: int) -> Series:
        """
        지수이동평균을 구하는 함수
        :param price: 지수이동평균을 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
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
        :param price: 가중이동평균을 구할 떄 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day:
        :return: N일간 price의 가중이동평균
        """
        weight = np.arange(1, day + 1)
        wma = price.rolling(window=day).apply(
            lambda prices: np.dot(prices, weight) / weight.sum(), raw=True
        )
        return wma

    @staticmethod
    def get_stddev(price: Series, day: int, ddof: int = 1) -> Series:
        return price.rolling(window=day).std(ddof=ddof)

    @classmethod
    def get_bollinger(
        cls,
        price: Series,
        day: int,
        r: int,
        moving_average: MovingAverage = MovingAverage.ewma,
    ) -> Tuple[Series, Series]:
        """
        Bollinger Bands를 구하는 함수
        Bollinger Bands의 상, 하한선은 표준 편차에 의해 산출된 이동평균 값이며,
        주가나 지수의 움직임이 큰 시기에는 Bands의 폭이 넓어지고 움직임이 작은 시기에는 Bands의 폭이 좁아지는 특정을 가지고 있다.
        따라서, 가격 움직임의 크기에 따라 밴드의 넓이가 결정된다.
        :param price: Bollinger Bands를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day:
        :param r: 표준편차 승수 ex) 2
        :param moving_average: 중간 밴드 기준 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: Bollinger Bands 상한선, 하한선
        """
        if moving_average == MovingAverage.sma:
            line_mid = cls.get_sma(price, day)
        elif moving_average == MovingAverage.ema:
            line_mid = cls.get_ema(price, day)
        elif moving_average == MovingAverage.ewma:
            line_mid = cls.get_ewma(price, day)
        elif moving_average == MovingAverage.wma:
            line_mid = cls.get_wma(price, day)
        line_std = cls.get_stddev(price, day, ddof=0)
        bollinger_range = line_std.multiply(r)
        upper = line_mid + bollinger_range
        lower = line_mid - bollinger_range
        return upper, lower

    @staticmethod
    def get_demark(
        price_open: Series, price_high: Series, price_low: Series, price_close: Series
    ) -> Tuple[Series, Series]:
        """
        Demark를 구하는 함수
        :param price_open: 시가
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :return: Demark 저항선, 지지선
        """
        d1 = np.where(
            price_close > price_open,
            (price_high.mul(2) + price_low + price_close) / 2,
            0,
        )
        d2 = np.where(
            price_close < price_open,
            (price_high + price_low.mul(2) + price_close) / 2,
            0,
        )
        d3 = np.where(
            price_close == price_open,
            (price_high + price_low + price_close.mul(2)) / 2,
            0,
        )
        d = Series(d1 + d2 + d3)
        demark_high = (d - price_low).shift(1)
        demark_low = (d - price_high).shift(1)

        return demark_high, demark_low

    @classmethod
    def get_envelope(
        cls,
        price: Series,
        day: int,
        r: float,
        moving_average: MovingAverage = MovingAverage.sma,
    ) -> Tuple[Series, Series]:
        """
        Envelope를 구하는 함수
        :param price: Envelope를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day:
        :param r: 비율 ex) 0.02, 0.08
        :param moving_average: 중심선 기준 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: Envelope 상한선, 하한선
        """
        if moving_average == MovingAverage.sma:
            line_mid = Indicator.get_sma(price, day)
        elif moving_average == MovingAverage.ema:
            line_mid = Indicator.get_ema(price, day)
        elif moving_average == MovingAverage.ewma:
            line_mid = Indicator.get_ewma(price, day)
        elif moving_average == MovingAverage.wma:
            line_mid = Indicator.get_wma(price, day)

        envelope_range = line_mid.multiply(r)
        upper = line_mid + envelope_range
        lower = line_mid - envelope_range
        return upper, lower

    @staticmethod
    def get_pivot(
        price_high: Series, price_low: Series, price_close: Series
    ) -> Tuple[Series, Series, Series, Series, Series]:
        """
        Pivot을 구하는 함수
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :return: 2차 저항선, 1차 저항선, Pivot 중심선, 1차 지지선, 2차 지지선
        """
        pivot = (price_high.shift(1) + price_low.shift(1) + price_close.shift(1)) / 3
        r2 = pivot + price_high.shift(1).sub(price_low.shift(1))
        r1 = (pivot * 2).sub(price_low.shift(1))
        s1 = (pivot * 2).sub(price_high.shift(1))
        s2 = pivot.sub(price_high.shift(1)) + price_low.shift(1)

        return r2, r1, pivot, s1, s2

    @staticmethod
    def get_price_channel(
        price_high: Series, price_low: Series, day: int
    ) -> Tuple[Series, Series]:
        """
        Price Channel를 구하는 함수
        Price Channel은 저항선과 지지선으로 구성된다.
        저항선은 일정기간전의 최고가를, 지지선은 일정기간전의 최저가를 이은선이다.
        이 채널은 채널포지션을 결정하는데 사용되지 않고 현재바의 최저가와 최고가를 보여줄 뿐이다.
        저항선은 뚜렷한 시장세력의 신호이며, 지지선은 뚜렷한 시장약화의 신호이다.
        :param price_high: 고가
        :param price_low: 저가
        :param day:
        :return: Price Channel 저항선, 지지선
        """
        resistance_line = price_high.shift(1).rolling(window=day).max()
        support_line = price_low.shift(1).rolling(window=day).min()
        return resistance_line, support_line

    @classmethod
    def get_dmi(
        cls, price_high: Series, price_low: Series, price_close: Series, day: int
    ) -> Tuple[Series, Series, Series]:
        """
        DMI를 구하는 함수
        DMI는 현재 시장의 추세와 강도를 함께 나타내는 지표로 단기보다는 중장기 추세분석에 유리하다.
        PDI는 실질적으로 상승하는 폭의 비율을 나타내며 MDI는 실질적으로 하락하는 폭의 비율을 의미한다.
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :param day:
        :return: +DI, -DI, ADX
        """
        data = {
            "range": abs(price_high - price_low),
            "up": abs(price_high - price_close.shift(1)),
            "down": abs(price_close.shift(1) - price_low),
        }
        data = DataFrame(data, columns=["range", "up", "down"])

        tr = data.max(axis=1)

        pdm = np.where(
            (
                (price_high.diff(1) > 0)
                & (price_high.diff(1) > price_low.shift(1) - price_low)
            ),
            price_high.diff(1),
            0,
        )
        pdmn = cls.get_ema(Series(pdm), day)
        mdm = np.where(
            (
                ((price_low.shift(1) - price_low) > 0)
                & (price_high.diff(1) < (price_low.shift(1) - price_low))
            ),
            price_low.shift(1) - price_low,
            0,
        )
        mdmn = cls.get_ema(Series(mdm), day)

        div = cls.get_ema(tr, day)

        pdi = pdmn.divide(div)
        mdi = mdmn.divide(div)
        dx = abs(pdi - mdi).divide(pdi + mdi)
        adx = cls.get_ema(dx, day)
        return pdi, mdi, adx

    @classmethod
    def get_macd(
        cls,
        price: Series,
        day_short: int,
        day_long: int,
        moving_average: MovingAverage = MovingAverage.ewma,
    ) -> Series:
        """
        MACD(Moving Average Convergence and Divergence)를 구하는 함수
        :param price: MACD를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day_short: 단기이동평균 기간
        :param day_long: 장기이동평균 기간
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: MACD
        """
        if moving_average == MovingAverage.sma:
            short_term = cls.get_sma(price, day_short)
            long_term = cls.get_sma(price, day_long)
        elif moving_average == MovingAverage.ema:
            short_term = cls.get_ema(price, day_short)
            long_term = cls.get_ema(price, day_long)
        elif moving_average == MovingAverage.ewma:
            short_term = cls.get_ewma(price, day_short)
            long_term = cls.get_ewma(price, day_long)
        elif moving_average == MovingAverage.wma:
            short_term = cls.get_wma(price, day_short)
            long_term = cls.get_wma(price, day_long)

        macd = short_term - long_term
        return macd

    @classmethod
    def get_macd_oscillator(
        cls,
        price: Series,
        day_short: int,
        day_long: int,
        day_signal: int,
        moving_average: MovingAverage = MovingAverage.ewma,
    ) -> Series:
        """
        MACD Oscillator를 구하는 함수
        MACD Oscillator는 MACD와 Signal의 교차를 보다 정확하게 인식하기 위해서 MACD 값에서 Signal 값을 뺀다.
        :param price: MACD Oscillator를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day_short: 단기이동평균 기간
        :param day_long: 장기이동평균 기간
        :param day_signal: 시그널 기간
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: MACD Oscillator
        """
        macd = cls.get_macd(price, day_short, day_long, moving_average)
        if moving_average == MovingAverage.sma:
            signal = cls.get_sma(macd, day_signal)
        elif moving_average == MovingAverage.ema:
            signal = cls.get_ema(macd, day_signal)
        elif moving_average == MovingAverage.ewma:
            signal = cls.get_ewma(macd, day_signal)
        elif moving_average == MovingAverage.wma:
            signal = cls.get_wma(macd, day_signal)

        macd_osillator = macd - signal
        return macd_osillator

    @staticmethod
    def get_momentum(price: Series, day: int) -> Series:
        """
        Momentum를 구하는 함수
        :param price: Momentum를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
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
        :param price: RSI를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day:
        :return: N일간 RSI
        """
        up = np.where(price.diff(1).gt(0), price.diff(1), 0)
        down = np.where(price.diff(1).lt(0), price.diff(1).multiply(-1), 0)
        average_up = Series(up).rolling(window=day, min_periods=day).mean()
        average_down = Series(down).rolling(window=day, min_periods=day).mean()
        return average_up.div(average_down + average_up)

    @classmethod
    def get_stochastic_fast(
        cls,
        price_high: Series,
        price_low: Series,
        price_close: Series,
        fast_k_period: int,
        fast_d_period: int,
        moving_average: MovingAverage = MovingAverage.ema,
    ) -> Tuple[Series, Series]:
        """
        Stochastic Fast 구하는 함수
        Stochastic은 적용기간 중에 움직인 가격 범위에서 오늘의 시장가격이 상대적으로 어디에 위치하고 있는지를 알려주는 지표로써,
        시장가격이 상승추세에 있다면 현재가격은 최고가 부근에 위치할 가능성이 높고,
        하락추세에 있다면 현재가는 최저가 부근에서 형성될 가능성이 높다는 것에 착안하여 만들어진 지표이다.
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :param fast_k_period: k기간
        :param fast_d_period: d기간
        :param moving_average: 이동평균선 종류 (일반적으로 지수이평선을 사용)
        :return: k_value, d_value (K: Stochastic Fast, D: Stochastic Fast를 이동평균한 값)
        """
        k_value = (price_close - price_low.rolling(window=fast_k_period).min()) / (
            price_high.rolling(window=fast_k_period).max()
            - price_low.rolling(window=fast_k_period).min()
        )
        if moving_average == MovingAverage.sma:
            d_value = cls.get_sma(k_value, fast_d_period)
        elif moving_average == MovingAverage.ema:
            d_value = cls.get_ema(k_value, fast_d_period)
        elif moving_average == MovingAverage.ewma:
            d_value = cls.get_ewma(k_value, fast_d_period)
        elif moving_average == MovingAverage.wma:
            d_value = cls.get_wma(k_value, fast_d_period)

        return k_value, d_value

    @staticmethod
    def get_volume_ratio(price_close: Series, volume: Series, day: int) -> Series:
        """
        Volume Ratio를 구하는 함수
        Volume Ratio는 일정 기간 동안 시장가격 상승일의 거래량과 시장가격 하락일의 거래량을 비교하여 나타낸 지표로서,
        현재 시장이 과열인지 침체인지를 판단하게 해주는 시장특성 지표이다.
        :param price_close: 종가
        :param volume: 거래량
        :param day:
        :return: N일간 Volume Ratio
        """
        up = np.where(price_close.diff(1).gt(0), volume, 0)
        down = np.where(price_close.diff(1).lt(0), volume, 0)
        maintain = np.where(price_close.diff(1).equals(0), volume.mul(0.5), 0)

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
        :param price: Psychological Line을 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
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
        :param price: New Psychological Line을 구할 떄 사용하는 가격 ex) price_open, price_high, price_low, price_close
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
    def get_disparity(
        cls, price: Series, day: int, moving_average: MovingAverage = MovingAverage.sma
    ) -> Series:
        """
        Disparity를 구하는 함수
        Disparity는 주가가 이동평균선과 어느 정도 떨어져 있는가를 분석한 것이다.
        :param price: Disparity를 구할 때 사용하는 가격 ex) price_open, price_high, price_low, price_close
        :param day:
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: N일간 이동평균과의 Disparity
        """
        if moving_average == MovingAverage.sma:
            moving_average = cls.get_sma(price, day)
        elif moving_average == MovingAverage.ema:
            moving_average = cls.get_ema(price, day)
        elif moving_average == MovingAverage.wma:
            moving_average = cls.get_wma(price, day)
        disparity = price.divide(moving_average)
        return disparity

    @staticmethod
    def get_ibs(price_high: Series, price_low: Series, price_close: Series) -> Series:
        """
        IBS를 구하는 함수
        IBS = (price_close - price_low) / (price_high - price_low)
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :return: IBS
        """
        return (price_close - price_low) / (price_high - price_low)

    @staticmethod
    def get_head_ratio(
        price_open: Series, price_high: Series, price_low: Series, price_close: Series
    ) -> Series:
        """
        윗꼬리 비율을 구하는 함수
        :param price_open: 시가
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :return: 윗꼬리 비율
        """

        price_upper = np.where(price_open >= price_open, price_open, price_close)
        price_upper = Series(price_upper)

        head_ratio = (price_high - price_upper) / (price_high - price_low)
        return head_ratio

    @staticmethod
    def get_tail_ratio(
        price_open: Series, price_high: Series, price_low: Series, price_close: Series
    ) -> Series:
        """
        아래꼬리 비율을 구하는 함수
        :param price_open: 시가
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :return: 아래꼬리 비율
        """

        price_under = np.where(price_open >= price_open, price_close, price_open)
        price_under = Series(price_under)

        tail_ratio = (price_under - price_low) / (price_high - price_low)
        return tail_ratio

    @classmethod
    def get_ab_ratio(
        cls,
        price_open: Series,
        price_high: Series,
        price_low: Series,
        price_close: Series,
        period: int,
    ) -> Tuple[Series, Series]:
        """
        AB Ratio를 구하는 함수
        AB Ratio는 주가 변동을 이용하여 강,약 에너지를 파악하고 이를 통해 주가의 움직임을 예측하는 지표이다.
        :param price_open: 시가
        :param price_high: 고가
        :param price_low: 저가
        :param price_close: 종가
        :param period: 기간
        :return: N일간의 AB Ratio
        """
        strength = price_high - price_open
        weakness = price_open - price_low
        a_ratio = (
            strength.rolling(window=period).sum()
            / weakness.rolling(window=period).sum()
        )

        strength = price_high - price_close.shift(1)
        weakness = price_close.shift(1) - price_low
        b_ratio = (
            strength.rolling(window=period).sum()
            / weakness.rolling(window=period).sum()
        )

        return a_ratio, b_ratio

    @classmethod
    def get_mass_index(
        cls,
        price_high: Series,
        price_low: Series,
        period: int,
        moving_average: MovingAverage = MovingAverage.ewma,
    ) -> Series:
        """
        Mass Index를 구하는 함수
        고가와 저가 사이의 변동폭을 측정하여 단기적인 추세의 전환점을 찾아내는 지표이다.
        Mass Index가 27선을 넘어선 후 26.5선을 하향 돌파하는 것을 reversal bulge라고 하는데,
        reversal bulge는 단기적인 추세의 전환을 암시한다.
        :param price_high: 고가
        :param price_low: 저가
        :param period: 합하고자 하는 일 수
        :param moving_average: 이동평균의 종류 (일반적으로 지수가중이동평균을 이용)
        :return: N일간 Mass Index
        """
        day_range = cls.get_range(price_high, price_low, 1)

        if moving_average == MovingAverage.sma:
            single = cls.get_sma(day_range, 9)
            double = cls.get_sma(single, 9)
        elif moving_average == MovingAverage.ema:
            single = cls.get_ema(day_range, 9)
            double = cls.get_ema(single, 9)
        elif moving_average == MovingAverage.ewma:
            single = cls.get_ewma(day_range, 9)
            double = cls.get_ewma(single, 9)
        elif moving_average == MovingAverage.wma:
            single = cls.get_wma(day_range, 9)
            double = cls.get_wma(single, 9)
        ratio = single / double

        return ratio.rolling(window=period).sum()

    @classmethod
    def get_range(
        cls,
        price_high: Series,
        price_low: Series,
        day: int,
        moving_average: MovingAverage = MovingAverage.sma,
    ) -> Series:
        """
        Range를 구하는 함수
        :param price_high: 고가
        :param price_low: 저가
        :param day:
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return: N일간 평균 Range
        """
        if moving_average == MovingAverage.sma:
            day_range = cls.get_sma(price_high - price_low, day)
        elif moving_average == MovingAverage.ema:
            day_range = cls.get_ema(price_high - price_low, day)
        elif moving_average == MovingAverage.ewma:
            day_range = cls.get_ewma(price_high - price_low, day)
        elif moving_average == MovingAverage.wma:
            day_range = cls.get_wma(price_high - price_low, day)
        return day_range

    @classmethod
    def get_mao(
        cls,
        price_close: Series,
        short_period: int,
        long_period: int,
        moving_average: MovingAverage = MovingAverage.sma,
    ) -> Series:
        """
        MAO를 구하는 함수
        MAO는 단기 이동 평균 값과 장기 이동 평균 값의 차이를 나타내어 주가 추세를 판단하기 위한 지표이다.
        :param price_close: 종가
        :param short_period: 단기 이동 평균 기간
        :param long_period: 장기 이동 평균 기간
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return:
        """
        if moving_average == MovingAverage.sma:
            mao = cls.get_sma(price_close, short_period) - Indicator.get_sma(
                price_close, long_period
            )
        elif moving_average == MovingAverage.ema:
            mao = cls.get_ema(price_close, short_period) - Indicator.get_ema(
                price_close, long_period
            )
        elif moving_average == MovingAverage.ewma:
            mao = cls.get_ewma(price_close, short_period) - Indicator.get_ewma(
                price_close, long_period
            )
        elif moving_average == MovingAverage.wma:
            mao = cls.get_wma(price_close, short_period) - Indicator.get_wma(
                price_close, long_period
            )
        return mao

    @classmethod
    def get_sonar(
        cls,
        price_close: Series,
        moving_average_period: int,
        sonar_period: int,
        sonar_moving_average_period: int,
        moving_average: MovingAverage = MovingAverage.ema,
    ) -> Tuple[Series, Series]:
        """
        sonar를 구하는 함수
        sonar는 주가의 추세 전환 시점을 파악하기 위한 지표이다.
        sonar는 가격 움직임에 선행하여 추세전환을 암시한다.
        예를 들어, 지표값이 0선을 상향 돌파할 때 매수시점으로, 0선을 하향돌파할 때 매도시점으로 인식한다.
        :param price_close: 종가
        :param moving_average_period: 이동평균 기간
        :param sonar_period: SONAR 기간
        :param sonar_moving_average_period: SONAR 이동평균 기간
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return:
        """
        if moving_average == MovingAverage.sma:
            sonar_moving_average = cls.get_sma(price_close, moving_average_period)
        elif moving_average == MovingAverage.ema:
            sonar_moving_average = cls.get_ema(price_close, moving_average_period)
        elif moving_average == MovingAverage.ewma:
            sonar_moving_average = cls.get_ewma(price_close, moving_average_period)
        elif moving_average == MovingAverage.wma:
            sonar_moving_average = cls.get_wma(price_close, moving_average_period)

        sonar = sonar_moving_average - sonar_moving_average.shift(sonar_period)

        if moving_average == MovingAverage.sma:
            signal = cls.get_sma(sonar, sonar_moving_average_period)
        elif moving_average == MovingAverage.ema:
            signal = cls.get_ema(sonar, sonar_moving_average_period)
        elif moving_average == MovingAverage.ewma:
            signal = cls.get_ewma(sonar, sonar_moving_average_period)
        elif moving_average == MovingAverage.wma:
            signal = cls.get_wma(sonar, sonar_moving_average_period)

        return sonar, signal

    @staticmethod
    def get_mfi(
        price_high: Series,
        price_low: Series,
        price_close: Series,
        vol: Series,
        day: int,
    ) -> Series:

        typical_price = (price_high + price_low + price_close) / 3
        money_flow = vol * typical_price

        positive_money_flow = np.where(typical_price.diff(1) > 0, money_flow, 0)
        positive_money_flow = (
            Series(positive_money_flow).rolling(window=day, min_periods=day).sum()
        )
        negative_money_flow = np.where(typical_price.diff(1) < 0, money_flow, 0)
        negative_money_flow = (
            Series(negative_money_flow).rolling(window=day, min_periods=day).sum()
        )

        money_flow_ratio = positive_money_flow / negative_money_flow
        mfi = money_flow_ratio / (1 + money_flow_ratio)

        return mfi

    @classmethod
    def get_trix(
        cls,
        price_close: Series,
        period: int,
        signal_period: int,
        moving_average: MovingAverage = MovingAverage.ema,
    ) -> Tuple[Series, Series]:
        """
        tirx를 구하는 함수
        :param price_close: 종가
        :param period: 이동평균 기간
        :param signal_period: 시그널 기간
        :param moving_average: 이동평균의 종류 ex) 단순이동평균, 지수이동평균, 가중이동평균
        :return:
        """
        if moving_average == MovingAverage.sma:
            ma3 = cls.get_sma(
                cls.get_sma(cls.get_sma(price_close, period), period), period
            )
        elif moving_average == MovingAverage.ema:
            ma3 = cls.get_ema(
                cls.get_ema(cls.get_ema(price_close, period), period), period
            )
        elif moving_average == MovingAverage.ewma:
            ma3 = cls.get_ewma(
                cls.get_ewma(cls.get_ewma(price_close, period), period), period
            )
        elif moving_average == MovingAverage.wma:
            ma3 = cls.get_wma(
                cls.get_wma(cls.get_wma(price_close, period), period), period
            )

        trix = ma3.diff(1) / ma3.shift(1)

        if moving_average == MovingAverage.sma:
            signal = cls.get_sma(trix, signal_period)
        elif moving_average == MovingAverage.ema:
            signal = cls.get_ema(trix, signal_period)
        elif moving_average == MovingAverage.ewma:
            signal = cls.get_ewma(trix, signal_period)
        elif moving_average == MovingAverage.wma:
            signal = cls.get_wma(trix, signal_period)

        return trix, signal
