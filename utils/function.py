from enum import Enum
from pandas import Series, DataFrame, merge
import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(x, y):
    linear = LinearRegression()
    model = linear.fit(x, y)
    coef = model.coef_
    return coef


class Method(Enum):
    average = 1
    min = 2
    max = 3
    first = 4
    dense = 5


class Function:

    @staticmethod
    def get_max(value: Series, day: int) -> Series:
        max_value = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).max())
        return Series(max_value)

    @staticmethod
    def get_min(value: Series, day: int) -> Series:
        min_value = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).min())
        return Series(min_value)

    @staticmethod
    def get_median(value: Series, day: int) -> Series:
        median = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).median())
        return median

    @staticmethod
    def get_argmax(value: Series, day: int) -> Series:
        argmax = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).argmax())
        return Series(argmax)

    @staticmethod
    def get_argmin(value: Series, day: int) -> Series:
        argmin = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).argmin())
        return Series(argmin)

    @staticmethod
    def get_quantile(value: Series, day: int, q: float) -> Series:
        quantile = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).quantile(q=q))
        return quantile

    @staticmethod
    def get_pct_change(value: Series, day: int) -> Series:
        pct_change = value.pct_change(periods=day)
        return pct_change

    @staticmethod
    def get_ts_rank(value: Series, day: int, method: Method = Method.average) -> Series:
        """
            method{‘average’, ‘min’, ‘max’, ‘first’, ‘dense’}, default ‘average’
            How to rank the group of records that have the same value (i.e. ties):
                average: average rank of the group
                min: lowest rank in the group
                max: highest rank in the group
                first: ranks assigned in order they appear in the array
                dense: like ‘min’, but rank always increases by 1 between groups.
        """
        ts_rank = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).rank(pct=True, method=method.name).iloc[-1])
        return ts_rank

    @staticmethod
    def get_ts_zscore(value: Series, day: int) -> Series:
        """
            using ddof = 0 or ddof = 1?
        """
        ts_zscore = value.rolling(window=day, min_periods=day).apply(
            lambda x: (Series(x).iloc[-1] - Series(x).mean()) / Series(x).std(ddof=1))
        return ts_zscore

    @staticmethod
    def get_ts_skew(value: Series, day: int) -> Series:
        ts_skew = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).skew())
        return ts_skew

    @staticmethod
    def get_ts_kurtosis(value: Series, day: int) -> Series:
        ts_kurtosis = value.rolling(window=day, min_periods=day).apply(
            lambda x: Series(x).kurtosis())
        return ts_kurtosis

    @staticmethod
    def get_ts_regression(value: Series, day: int) -> Series:
        ts_regression = value.rolling(window=day, min_periods=day).apply(
            lambda x: linear_regression([[x_val] for x_val in np.arange(1, day + 1)],
                                        [[(y_val - Series(x).values[0]) / Series(x).values[0] * 100]
                                         for y_val in Series(x).values]))
        return ts_regression

    @staticmethod
    def get_ts_correlation(lhs_value: DataFrame, rhs_value: DataFrame, window_size: int):
        merged = merge(left=lhs_value, right=rhs_value, how='outer', on='date', sort=True, suffixes=["-lhs", "-rhs"])
        total_value = DataFrame(merged["date"])

        lhs_roll = merged['close-lhs'].rolling(window_size)
        rhs_roll = merged['close-rhs'].rolling(window_size)
        rhs_iterator = iter(rhs_roll)
        correlation_list = list()
        for lhs in lhs_roll:
            rhs = next(rhs_iterator)
            correlation_list.append(lhs.corr(rhs))

        total_value['correlation'] = correlation_list

        lhs = merge(lhs_value, total_value, on='date')
        rhs = merge(rhs_value, total_value, on='date')
        return lhs, rhs

    @staticmethod
    def get_correlation():
        pass

    @staticmethod
    def truncate(value: Series, turning_value: int) -> Series:
        return value.where(value > turning_value)

    @staticmethod
    def backfill(value: Series, change_value: int or float) -> Series:
        return value.fillna(change_value)

    @staticmethod
    def ts_backfill(value: Series) -> Series:
        ts_backfill = value.fillna(method='ffill')
        return ts_backfill
