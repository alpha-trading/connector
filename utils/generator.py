from typing import Optional

from utils import Executor
from datetime import date
from pandas import DataFrame

TABLE_RAW_CANDLE_DAY = 'data_candleday'
TABLE_EDITED_CANDLE_DAY = 'data_editedcandleday'


class Generator:
    def __init__(self, executor: Executor):
        self.executor = executor

    def get_trading_day_list(self, start_date: date, end_date: date) -> list:
        """
        과거 거래일 리스트를 조회하는 함수
        :param start_date: back test 시작일
        :param end_date: back test 종료일
        :return: 과거 거래일 리스트 반환
        """
        query = f"SELECT * FROM data_iskoreatradingday " \
                f"WHERE is_tradable = '1' and date >= '{start_date}' and date <= '{end_date}'"
        df = self.executor.sql(query)
        trading_day_list = df['date'].to_list()
        return trading_day_list

    def get_past_universe_stock_list(self, universe: str) -> list:
        """
        과거에 한번이라도 universe에 포함된 종목 리스트 반환
        :param universe: kospi200 또는 kosdaq150
        :return: universe에 포함되었던 종목 리스트 반환
        """
        query = f"SELECT ticker FROM data_{universe} GROUP BY ticker"
        df = self.executor.sql(query)
        past_universe_stock_list = df['ticker'].to_list()
        return past_universe_stock_list

    def _get_day_price_data(self, universe: str, table_name: str) -> dict:
        day_price = {}
        past_universe_stock_list = self.get_past_universe_stock_list(universe)
        for ticker in past_universe_stock_list:
            query = f"SELECT candle.*, ticker.ticker FROM {table_name} " \
                    f"AS candle INNER JOIN data_ticker AS ticker ON ticker.id = candle.ticker_id " \
                    f"WHERE ticker = '{ticker}' and market = '{universe[:-3]}'"
            df = self.executor.sql(query)
            df = df.drop(['id', 'ticker_id', 'ticker'], axis=1)
            day_price[ticker] = df

        return day_price

    def get_day_price_data(self, universe: str) -> dict:
        """
        universe에 포함된 종목들의 무수정 주가 일봉 데이터 반환
        :param universe: kospi200 또는 kosdaq150
        :return: 일봉 데이터
        """
        return self._get_day_price_data(universe, TABLE_RAW_CANDLE_DAY)

    def get_edited_day_price_data(self, universe: str) -> dict:
        """
        universe에 포함된 종목들의 수정주가 일봉 데이터 반환
        :param universe:
        :return: 일봉 데이터
        """
        return self._get_day_price_data(universe, TABLE_EDITED_CANDLE_DAY)

    def get_today_universe_stock_list(self, universe: str, today: date) -> list:
        """
        당일의 universe에 포함된 종목 리스트 반환
        :param universe: kospi200 또는 kosdaq150
        :param today: 당일 날짜
        :return: 해당 universe의 종목 리스트
        """

        query = f"SELECT * FROM data_{universe}history where date = '{today}'"
        df = self.executor.sql(query)
        today_universe_stock_list = df['tickers'].iloc[0].split(',')

        return today_universe_stock_list

    def get_index(self, universe: str) -> Optional[DataFrame]:
        """
        해당 universe의 index 가격을 조회
        :param universe: kospi200 또는 kosdaq150
        :return:
        """
        if universe == 'kospi200':
            ticker = '069500'
        elif universe == 'kosdaq150':
            ticker = '229200'
        else:
            return None

        query = f"SELECT candle.*, ticker.ticker FROM data_candleday " \
                f"AS candle INNER JOIN data_ticker AS ticker ON ticker.id = candle.ticker_id " \
                f"WHERE ticker = '{ticker}'"
        df = self.executor.sql(query)
        df = df.drop(['id', 'ticker_id', 'ticker'], axis=1)

        return df

    def get_won_dollar_exchange_rate(self) -> DataFrame:
        """
        원 달러 환율 조회
        :return: 원 달러 환율 반환
        """
        query = f"SELECT * FROM data_wondollarexchangerate"
        df = self.executor.sql(query)

        return df
