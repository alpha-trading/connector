from pandas import Series, DataFrame, merge
import numpy as np
from sklearn.linear_model import LinearRegression

from utils.parameter import KindOfAverage


def _linear_regression(x, y):
    linear = LinearRegression()
    model = linear.fit(x, y)
    coef = model.coef_
    return coef


def rank(value: Series) -> Series:
    """
    순위

    <설명>
    당일 코스피, 코스닥 구성 종목에서 해당 종목의 순위를 반환하는 함수입니다.
    0과 1사이의 값을 가지며, 1에 가까울수록 순위가 높다는 의미입니다.

    <사용 방법>
    첫 번째 인자에 순위를 구하고자 하는 값을 적으면 됩니다.
    예를 들어, 20일 평균 거래대금의 순위를 구하고자 하는 경우에는
    'rank(sma(tr_val, 20))' 또는 '순위(단순이동평균(거래대금, 20))'과 같이 작성하면 됩니다.

    :param value: 순위를 구하고자 하는 값
    :return:
    """
    return value.rank(pct=True)


def ts_sum(value: Series, period: int) -> Series:
    """
    기간합계

    <설명>
    기간합계(ts_sum)는 특정 기간(period)동안 값(value)들의 합을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간합계(ts_sum)를 구하고자 하는 값을,
    두 번째 인자는 기간합계(ts_sum)를 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 5일간 거래량의 합을 구하고자 하는 경우에는
    'ts_sum(volume, 5) 또는 '기간합계(거래량, 5)'와 같이 작성하면 됩니다.

    :param value: 기간합계(ts_sum)를 구하고자 하는 값
    :param period: 기간합계(ts_sum)를 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).sum()


def ts_max(value: Series, period: int) -> Series:
    """
    기간최댓값

    <설명>
    기간최댓값(ts_max)은 특정 기간(period)동안 값(value)들 중 최댓값을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간최댓값(ts_max)을 구하고자 하는 값을,
    두 번째 인자는 기간최댓값(ts_max)을 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 최근 5일간 종가 중 최댓값을 구하고자 하는 경우에는
    'ts_max(close, 5) 또는 '기간최댓값(종가, 5)'와 같이 작성하면 됩니다.

    :param value: 기간최댓값(ts_max)을 구하고자 하는 값
    :param period: 기간최댓값(ts_max)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).max()


def ts_min(value: Series, period: int) -> Series:
    """
    기간최솟값

    <설명>
    기간최솟값(ts_min)은 특정 기간(period)동안 값(value)들 중 최솟값을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간최솟값(ts_min)을 구하고자 하는 값을,
    두 번째 인자는 기간최솟값(ts_min)을 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 최근 5일간 종가 중 최솟값을 구하고자 하는 경우에는
    'ts_min(close, 5) 또는 '기간최솟값(종가, 5)'와 같이 작성하면 됩니다.

    :param value: 기간최솟값(ts_min)을 구하고자 하는 값
    :param period: 기간최솟값(ts_min)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).min()


def ts_median(value: Series, period: int) -> Series:
    """
    기간중앙값

    <설명>
    기간중앙값(ts_median)은 특정 기간(period)동안 값(value)들 중 중앙값을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간중앙값(ts_median)을 구하고자 하는 값을,
    두 번째 인자는 기간중앙값(ts_median)을 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 최근 5일간 종가 중에서 중앙값을 구하고자 하는 경우에는
    'ts_median(close, 5) 또는 '기간중앙값(종가, 5)'와 같이 작성하면 됩니다.

    :param value: 기간중앙값(ts_median)을 구하고자 하는 값
    :param period: 기간중앙값(ts_median)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).median()


def ts_argmax(value: Series, period: int) -> Series:
    """
    기간최대일

    <설명>
    기간최대일(ts_argmax)은 특정 기간(period)동안 값(value)들 중 최댓값이 되는 날을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간최대일(ts_argmax)을 구하고자 하는 값을,
    두 번째 인자는 기간최대일(ts_argmax)을 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 최근 5일간 종가의 기간최대일(ts_argmax)을 구하고자 하는 경우에는
    'ts_argmax(close, 5) 또는 '기간최대일(종가, 5)'와 같이 작성하면 됩니다.

    :param value: 기간최대일(ts_argmax)을 구하고자 하는 값
    :param period: 기간최대일(ts_argmax)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).apply(lambda x: Series(x).argmax())


def ts_argmin(value: Series, period: int) -> Series:
    """
    기간최소일

    <설명>
    기간최소일(ts_argmin)은 특정 기간(period)동안 값(value)들 중 최솟값이 되는 날을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간최소일(ts_argmin)을 구하고자 하는 값을,
    두 번째 인자는 기간최소일(ts_argmin)을 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 최근 5일간 종가의 기간최소일(ts_argmin)을 구하고자 하는 경우에는
    'ts_argmin(close, 5) 또는 '기간최소일(종가, 5)'와 같이 작성하면 됩니다.

    :param value: 기간최소일(ts_argmin)을 구하고자 하는 값
    :param period: 기간최소일(ts_argmin)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).apply(lambda x: Series(x).argmin())


def ts_quantile(value: Series, period: int, percentile: float) -> Series:
    """
    기간백분위수

    <설명>
    기간백분위수(ts_quantile)는 특정 기간(period)동안의 값(value)들 중 특정 백분위(percentile)에 해당하는 값을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간백분위수(ts_quantile)을 구하고자 하는 값을,
    두 번째 인자는 기간백분위수(ts_quantile)을 구하는데 사용하고자 하는 기간,
    세 번째 인자는 기간백분위수(ts_quantile)을 구하는데 사용하는 백분위를 적으면 됩니다.
    예를 들어, 최근 20일간 종가 중에서 75%에 해당하는 값을 구하고자 하는 경우에는
    'ts_quantile(close, 20, 0.75) 또는 '기간백분위수(종가, 20, 0.75)'와 같이 작성하면 됩니다.

    :param value: 기간백분위수(ts_quantile)를 구하고자 하는 값
    :param period: 기간백분위수(ts_quantile)를 구하는데 사용하고자 하는 기간
    :param percentile: 기간백분위수(ts_quantile)를 구하는데 사용하는 백분위 값
    :return:
    """
    return value.rolling(window=period, min_periods=period).quantile(quantile=percentile)


def ts_stddev(value: Series, period: int) -> Series:
    """
    기간표준편차

    <설명>
    기간표준편차(ts_stddev)는 특정 기간(period)동안 값(value)들의 표준편차를 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간표준편차(ts_stddev)를 구하고자 하는 값을,
    두 번째 인자는 기간표준편차(ts_stddev)를 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 20일간 종가의 표준편차를 구하고자 하는 경우에는
    'ts_stddev(close, 20) 또는 '기간표준편차(종가, 20)'과 같이 작성하면 됩니다.

    :param value: 기간표준편차(ts_stddev)를 구하고자 하는 값
    :param period: 기간표준편차(ts_stddev)를 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period).std(ddof=1)


def pct_change(value: Series, period: int) -> Series:
    """
    변화율

    <설명>
    변화율(pct_change)은 특정 기간(period)동안 값(value)의 변화율을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 변화율(pct_change)을 구하고자 하는 값을,
    두 번째 인자는 변화율(pct_change)을 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 20일간 종가의 변화율 즉 20일 모멘텀을 구하고자 하는 경우에는
    'pct_change(close, 20) 또는 '변화율(종가, 20)'과 같이 작성하면 됩니다.

    :param value: 변화율(pct_change)을 구하고자 하는 값
    :param period: 변화율(pct_change)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.pct_change(periods=period)
    # 정확한 시간 측정을 통해 빠른 연산을 선택해야
    # return value / value.shift(period) - 1


def sigmoid(value: Series) -> Series:
    """
    시그모이드

    <설명>
    값들을 정규화하는 함수 중 하나입니다.
    시그모이드(sigmoid)를 사용하면 값들을 0과 1사이로 정규화시킬 수 있습니다.

    <사용 방법>
    첫 번째 인자에 정규화시키고자하는 값을 적으면 됩니다.
    예를 들어, 5일 종가 평균을 시그모이드 함수를 이용해 0과 1사이의 값으로 만들고자하는 경우에는
    'sigmoid(sma(close, 5)) 또는 '시그모이드(단순이동평균(종가, 5))'와 같이 작성하면 됩니다.

    :param value: 정규화시키고자하는 값
    :return:
    """
    return 1 / (1 + np.exp(-value))


def tanh(value: Series) -> Series:
    """
    하이퍼블릭탄젠트

    <설명>
    값들을 정규화하는 함수 중 하나입니다.
    하이퍼블릭탄젠트(tanh)를 사용하면 값들을 0과 1사이로 정규화시킬 수 있습니다.

    <사용 방법>
    첫 번째 인자에 정규화시키고자하는 값을 적으면 됩니다.
    예를 들어, 5일 종가 평균을 시그모이드 함수를 이용해 0과 1사이의 값으로 만들고자하는 경우에는
    'sigmoid(sma(close, 5)) 또는 '시그모이드(단순이동평균(종가, 5))'와 같이 작성하면 됩니다.

    :param value: 정규화시키고자하는 값
    :return:
    """
    return np.tanh(value)


def ts_rank(value: Series, period: int) -> Series:
    """
    기간순위

    <설명>
    한 종목의 당일 값의 순위를 특정 기간동안의 값들 중에서 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간순위(ts_rank)를 구하고자 하는 값을,
    두 번째 인자는 기간순위(ts_rank)를 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 60일간 종가의 기간순위를 구하고자 하는 경우에는
    'ts_rank(close, 60) 또는 '기간순위(종가, 60)'과 같이 작성하면 됩니다.

    :param value: 기간순위(ts_rank)를 구하고자 하는 값
    :param period: 기간순위(ts_rank)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).apply(
        lambda x: Series(x).rank(pct=True, method=KindOfAverage.average.name).iloc[-1]
    )


def ts_zscore(value: Series, period: int) -> Series:
    """
    기간표준점수

    <설명>
    한 종목의 당일 표준점수를 특정 기간동안의 값들을 이용해서 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간표준점수(ts_zscore)를 구하고자 하는 값을,
    두 번째 인자는 기간표준점수(ts_zscore)를 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 60일간 종가의 기간표준점수(ts_zscore)를 구하고자 하는 경우에는
    'ts_zscore(close, 60) 또는 '기간표준점수(종가, 60)'과 같이 작성하면 됩니다.

    :param value: 기간표준점수(ts_zscore)를 구하고자 하는 값
    :param period: 기간표준점수(ts_zscore)을 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).apply(
        lambda x: (Series(x).iloc[-1] - Series(x).mean()) / Series(x).std(ddof=1)
    )


def ts_skew(value: Series, period: int) -> Series:
    """
    기간왜도

    <설명>
    기간왜도(ts_skew)는 특정 기간(period)동안 값(value)들의 왜도(skewness) 값을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간왜도(ts_skew)를 구하고자 하는 값을,
    두 번째 인자는 기간왜도(ts_skew)를 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 20일간 종가의 왜도 값을 구하고자 하는 경우에는
    'ts_skew(close, 20) 또는 '기간왜도(종가, 20)'과 같이 작성하면 됩니다.

    :param value: 기간왜도(ts_skew)를 구하고자 하는 값
    :param period: 기간왜도(ts_skew)를 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).skew()


def ts_kurt(value: Series, period: int) -> Series:
    """
    기간첨도

    <설명>
    기간첨도(ts_kurt)는 특정 기간(period)동안 값(value)들의 첨도(kurtosis) 값을 구하는 함수입니다.

    <사용 방법>
    첫 번째 인자는 기간첨도(ts_kurt)를 구하고자 하는 값을,
    두 번째 인자는 기간첨도(ts_kurt)를 구하는데 사용하고자 하는 기간을 적으면 됩니다.
    예를 들어, 20일간 종가의 왜도 값을 구하고자 하는 경우에는
    'ts_kurt(close, 20) 또는 '기간첨도(종가, 20)'과 같이 작성하면 됩니다.

    :param value: 기간첨도(ts_kurt)를 구하고자 하는 값
    :param period: 기간첨도(ts_kurt)를 구하는데 사용하고자 하는 기간
    :return:
    """
    return value.rolling(window=period, min_periods=period).kurt()


def ts_regression(value: Series, period: int) -> Series:
    return value.rolling(window=period, min_periods=period).apply(
        lambda x: _linear_regression(
            [[x_val] for x_val in np.arange(1, period + 1)],
            [[(y_val - Series(x).values[0]) / Series(x).values[0] * 100] for y_val in Series(x).values],
        )
    )


def get_ts_correlation(lhs_value: DataFrame, rhs_value: DataFrame, window_size: int):
    merged = merge(left=lhs_value, right=rhs_value, how="outer", on="date", sort=True, suffixes=["-lhs", "-rhs"])
    total_value = DataFrame(merged["date"])

    lhs_roll = merged["close-lhs"].rolling(window_size)
    rhs_roll = merged["close-rhs"].rolling(window_size)
    rhs_iterator = iter(rhs_roll)
    correlation_list = list()
    for lhs in lhs_roll:
        rhs = next(rhs_iterator)
        correlation_list.append(lhs.corr(rhs))

    total_value["correlation"] = correlation_list

    lhs = merge(lhs_value, total_value, on="date")
    rhs = merge(rhs_value, total_value, on="date")
    return lhs, rhs
