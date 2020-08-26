from datetime import datetime
from pandas import DataFrame, Series

from utils.tools import year_to_date


class Evaluator:
    @staticmethod
    def _get_profit(pnl: DataFrame, pnl_column='pnl') -> Series:
        """거래 차익 컬럼 추가"""
        profit = pnl[pnl_column].diff()
        return profit

    @staticmethod
    def _get_profit_rate(pnl: DataFrame, pnl_column='pnl', profit_column='profit') -> Series:
        """이익율 컬럼 추가"""
        profit_rate = pnl[profit_column] / pnl.shift(1)[pnl_column]
        return profit_rate

    @staticmethod
    def _get_cumulative_yield(pnl: DataFrame) -> float:
        """누적수익율"""
        cumulative_yield = (pnl['pnl'].iloc[-1] - pnl['pnl'].iloc[0]) / pnl['pnl'].iloc[0]
        return cumulative_yield

    @staticmethod
    def _get_winning_rate(pnl: DataFrame) -> (float, int):
        """승률과 거래일을 구함"""
        is_win = pnl.profit > 0

        winning_rate = is_win.mean()
        trading_days = is_win.count()
        return winning_rate, trading_days

    @staticmethod
    def _get_profit_loss_rate(pnl: DataFrame) -> float:
        """손익비"""
        profit_pnl = pnl[pnl.profit_rate > 0]
        loss_pnl = pnl[pnl.profit_rate < 0]
        rate = (profit_pnl['profit_rate'].sum() / len(profit_pnl)) / (
            abs(loss_pnl['profit_rate'].sum()) / len(loss_pnl))
        return rate

    @staticmethod
    def _get_mdd(pnl: DataFrame) -> Series:
        """최대 낙폭"""
        ath = pnl.rolling(len(pnl), min_periods=1)['pnl'].max()
        dd = pnl['pnl'] - ath
        mdd = dd.rolling(len(dd), min_periods=1).min() / ath

        return mdd

    @staticmethod
    def _get_shape_ratio(pnl: DataFrame) -> float:
        """샤프 지수"""
        return pnl['profit_rate'].mean() / pnl['profit_rate'].std()

    @staticmethod
    def _remove_non_trading_days(pnl: DataFrame) -> DataFrame:
        """거래가 일어나지 않은 날짜 제거"""
        pnl = pnl[pnl['profit'] != 0]
        return pnl

    @classmethod
    def pnl(cls, pnl: DataFrame) -> DataFrame:
        """
        PNL 받아서 알파 성능 측정
        :param pnl: DataFrame columns: date, pnl
        :return: 거래일수, 승률, 손익비, 샤프지수, 최대 낙폭 퍼센트, 누적수익율
        """

        first = pnl.iloc[0].name  # type: datetime
        last = pnl.iloc[-1].name  # type: datetime

        # 끝에 잘라냄

        result = {}

        pnl['profit'] = cls._get_profit(pnl)
        pnl['profit_rate'] = cls._get_profit_rate(pnl)
        pnl['mdd'] = cls._get_mdd(pnl)
        pnl = pnl[1:]
        raw_pnl = pnl.copy()
        pnl = cls._remove_non_trading_days(pnl)

        for year in range(first.year, last.year + 1):
            start = year_to_date(year)
            end = year_to_date(year + 1)
            mask = (pnl.index >= start) & (pnl.index < end)
            this_year_df = pnl[mask]
            winning_rate, trading_days = cls._get_winning_rate(this_year_df)
            mdd = this_year_df['mdd'].min()
            year_dict = {
                'trading_days': trading_days,
                'winning_rate': winning_rate,
                'profit_loss_rate': cls._get_profit_loss_rate(this_year_df),
                'shape_ratio': cls._get_shape_ratio(this_year_df),
                'mdd': mdd,
                'cumulative_yield': cls._get_cumulative_yield(this_year_df),
            }
            result[year] = year_dict

        winning_rate, trading_days = cls._get_winning_rate(pnl)
        mdd = pnl['mdd'].min()
        result['total'] = {
            'trading_days': trading_days,
            'winning_rate': winning_rate,
            'profit_loss_rate': cls._get_profit_loss_rate(pnl),
            'shape_ratio': cls._get_shape_ratio(raw_pnl),
            'mdd': mdd,
            'cumulative_yield': cls._get_cumulative_yield(raw_pnl),
        }
        year_df = DataFrame.from_dict(result, orient='index')
        return year_df
