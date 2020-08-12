from datetime import datetime
from pandas import DataFrame

from utils.tools import year_to_datetime


class Evaluator:
    @staticmethod
    def _add_profit(pnl: DataFrame) -> DataFrame:
        """거래 차익 컬럼 추가"""
        with_profit = pnl.copy()
        with_profit['profit'] = pnl.pnl.diff().shift(1)
        return with_profit

    @staticmethod
    def _add_profit_rate(pnl: DataFrame) -> DataFrame:
        """이익율 컬럼 추가"""
        with_profit = pnl.copy()
        with_profit['profit_rate'] = pnl.profit / pnl.shift(1).pnl * 100
        return with_profit

    @staticmethod
    def _get_winning_rate(pnl: DataFrame) -> (float, int):
        """승률과 거래일을 구함"""
        with_win = pnl.copy()
        with_win['is_win'] = pnl.profit > 0

        winning_rate = with_win['is_win'].mean()
        trading_days = with_win['is_win'].count()
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
    def _get_shape_ratio(pnl: DataFrame) -> float:
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
        :return: 거래일수, 승률, 손익비, 샤프지수
        """

        ### 수정 사항 ---> 연도 별로 pnl 평가하도록 하자...! (total과 병행....!)

        first = pnl.iloc[0].name  # type: datetime
        last = pnl.iloc[-1].name  # type: datetime

        # 끝에 잘라냄

        result = {}

        pnl = cls._add_profit(pnl)
        pnl = cls._add_profit_rate(pnl)
        pnl = pnl[1:]
        raw_pnl = pnl.copy()
        pnl = cls._remove_non_trading_days(pnl)

        for year in range(first.year, last.year + 1):
            start = year_to_datetime(year)
            end = year_to_datetime(year + 1)
            mask = (pnl.index >= start) & (pnl.index < end)
            this_year_df = pnl[mask]
            winning_rate, trading_days = cls._get_winning_rate(this_year_df)
            year_dict = {
                'trading_days': trading_days,
                'winning_rate': winning_rate,
                'profit_loss_rate': cls._get_profit_loss_rate(this_year_df),
                'shape_ratio': cls._get_shape_ratio(this_year_df),
            }
            result[year] = year_dict

        winning_rate, trading_days = cls._get_winning_rate(pnl)
        result['total'] = {
            'trading_days': trading_days,
            'winning_rate': winning_rate,
            'profit_loss_rate': cls._get_profit_loss_rate(pnl),
            'shape_ratio': cls._get_shape_ratio(raw_pnl),
        }
        year_df = DataFrame.from_dict(result, orient='index')
        return year_df
