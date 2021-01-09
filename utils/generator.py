from pypika import MySQLQuery, Table, Tables, Criterion
from typing import Optional
from enum import Enum

from utils import Executor
from datetime import date
from pandas import DataFrame

TABLE_RAW_CANDLE_DAY = 'data_candleday'
TABLE_EDITED_CANDLE_DAY = 'data_editedcandleday'


class ExchangeRateEnum(Enum):
    dollar = 'USDKRW'
    euro = 'EURKRW'
    yen = 'JPYKRW'


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
        data_trading_day = Table('data_iskoreatradingday')
        query = MySQLQuery.from_(data_trading_day).select(data_trading_day.date).where(Criterion.all([
                data_trading_day.is_tradable == '1',
                data_trading_day.date >= start_date,
                data_trading_day.date <= end_date
            ]))

        df = self.executor.sql(query.get_sql())
        trading_day_list = df['date'].to_list()
        return trading_day_list

    def get_past_universe_stock_list(self, universe: str) -> list:
        """
        과거에 한번이라도 universe에 포함된 종목 리스트 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :return: universe에 포함되었던 종목 리스트 반환
        """
        if universe in ('total', 'kospi', 'kosdaq'):
            data_ticker = Table('data_ticker')
            if universe == 'total':
                query = MySQLQuery.from_(data_ticker).groupby(data_ticker.ticker).select('*')
            else:
                query = MySQLQuery.from_(data_ticker).groupby(data_ticker.id).having(
                    data_ticker.market == universe).select('*')
            df = self.executor.sql(query.get_sql())
            past_universe_stock_list = df['ticker'].to_list()

            return past_universe_stock_list

        elif universe in ('top350', 'kospi200', 'kosdaq150'):
            if universe == 'top350':
                kospi200_past_universe_stock_list = self.get_past_universe_stock_list('kospi200')
                kosdaq150_past_universe_stock_list = self.get_past_universe_stock_list('kosdaq150')
                return kospi200_past_universe_stock_list + kosdaq150_past_universe_stock_list
            elif universe in ('kospi200', 'kosdaq150'):
                data_kospi200 = Table(f'data_{universe}')
                query = MySQLQuery.from_(data_kospi200).groupby(data_kospi200.ticker_id).select(
                    data_kospi200.ticker_id)
                df = self.executor.sql(query.get_sql())
                past_universe_stock_list = df['ticker_id'].to_list()
                return past_universe_stock_list

    def __get_day_price_data(self, universe: str, table_name: str, trading_share: bool, trading_trend: bool,
                             start_date: date, end_date: date) -> dict:

        past_universe_stock_list = self.get_past_universe_stock_list(universe)
        past_universe_stock_list = tuple(past_universe_stock_list)
        data_day_price, data_ticker = Tables(table_name, 'data_ticker')

        query = MySQLQuery.from_(data_day_price).join(data_ticker).on(data_day_price.ticker_id == data_ticker.id)
        if trading_share is True:
            data_share = Table('data_daytradinginfo')
            query = query.join(data_share).on(Criterion.all([
                data_day_price.ticker_id == data_share.ticker_id,
                data_day_price.date == data_share.date]))
        if trading_trend is True:
            data_trend = Table('data_daytradingtrend')
            query = query.join(data_trend).on(Criterion.all([
                data_day_price.ticker_id == data_share.ticker_id,
                data_day_price.date == data_share.date
            ]))
        if trading_share is False and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker)
        if trading_share is True and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker, data_share.cap, data_share.shares_out)
        if trading_share is False and trading_trend is True:
            pass
        if trading_share is True and trading_trend is True:
            pass
        query = query.where(Criterion.all([
            data_ticker.ticker.isin(past_universe_stock_list),
            data_day_price.date >= start_date,
            data_day_price.date <= end_date
        ]))
        df = self.executor.sql(query.get_sql())
        df = df.drop(['id'], axis=1)
        return df

    def get_day_price_data(self, universe: str, trading_share: bool, trading_trend: bool,
                           start_date: date, end_date: date) -> dict:

        """
        universe에 포함된 종목들의 무수정 주가 일봉 데이터 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :param trading_share: 상장주식수, 상장시가총액 데이터 포함여부
        :param trading_trend: 기간 순매수량, 외국인 순매수량 데이터 등 포함여부
        :param start_date:
        :param end_date:
        :return: 일봉 데이터
        """
        return self.__get_day_price_data(universe, TABLE_RAW_CANDLE_DAY, trading_share, trading_trend,
                                         start_date, end_date)

    def get_edited_day_price_data(self, universe: str) -> dict:
        """
        universe에 포함된 종목들의 수정주가 일봉 데이터 반환
        :param universe:
        :return: 일봉 데이터
        """
        return self.__get_day_price_data(universe, TABLE_EDITED_CANDLE_DAY)

    def __get_day_price_data_by_ticker(self, ticker, table_name, trading_share: bool, trading_trend: bool,
                                       start_date: date, end_date: date):

        data_day_price, data_ticker = Tables(table_name, 'data_ticker')

        query = MySQLQuery.from_(data_day_price).join(data_ticker).on(data_day_price.ticker_id == data_ticker.id)
        if trading_share is True:
            data_share = Table('data_daytradinginfo')
            query = query.join(data_share).on(Criterion.all([
                data_day_price.ticker_id == data_share.ticker_id,
                data_day_price.date == data_share.date]))
        if trading_trend is True:
            data_trend = Table('data_daytradingtrend')
            query = query.join(data_trend).on(Criterion.all([
                data_day_price.ticker_id == data_share.ticker_id,
                data_day_price.date == data_share.date
            ]))
        if trading_share is False and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker)
        if trading_share is True and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker, data_share.cap, data_share.shares_out)
        if trading_share is False and trading_trend is True:
            pass
        if trading_share is True and trading_trend is True:
            pass
        query = query.where(Criterion.all([
                data_ticker.ticker == ticker,
                data_day_price.date >= start_date,
                data_day_price.date <= end_date
            ]))
        df = self.executor.sql(query.get_sql())
        df = df.drop(['id'], axis=1)
        return df

    def get_day_price_data_by_ticker(self, ticker, trading_share: bool, trading_trend: bool,
                                     start_date: date, end_date: date):
        """
        universe에 포함된 종목들의 수정 주가 일봉 데이터 반환
        :param ticker: 개별 종목의 ticker
        :param trading_share:
        :param trading_trend:
        :param start_date:
        :param end_date:
        :return: 해당 ticker의 일봉 데이터
        """
        return self.__get_day_price_data_by_ticker(ticker, TABLE_RAW_CANDLE_DAY, trading_share, trading_trend,
                                                   start_date, end_date)

    def get_edited_day_price_data(self, ticker):
        """
        universe에 포함된 종목들의 수정 주가 일봉 데이터 반환
        :param ticker: 개별 종목의 ticker
        :return: 해당 ticker의 일봉 데이터
        """
        return self.__get_edited_day_price_data_by_ticker(ticker, TABLE_EDITED_CANDLE_DAY)

    def get_today_universe_stock_list(self, universe: str, today: date) -> list:
        """
        당일의 universe에 포함된 종목 리스트 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :param today: 당일 날짜
        :return: 해당 universe의 종목 리스트
        """
        if universe in ('kospi', 'kosdaq', 'kospi200', 'kosdaq150'):
            data_universe = Table(f'data_{universe}history')
            query = MySQLQuery.from_(data_universe).select(data_universe.tickers).where(
                data_universe.date == today)
            df = self.executor.sql(query.get_sql())
            today_universe_stock_list = df['tickers'].iloc[0].split(',')
        elif universe in ('total', 'top350'):
            if universe == 'total':
                today_universe_stock_list = self.get_today_universe_stock_list('kospi', today) + \
                                            self.get_today_universe_stock_list('kosdaq', today)
            elif universe == 'top350':
                today_universe_stock_list = self.get_today_universe_stock_list('kospi200', today) + \
                                            self.get_today_universe_stock_list('kosdaq150', today)

        return today_universe_stock_list

    def get_index_day_price_data(self, universe: str, start_date: date, end_date: date) -> Optional[DataFrame]:
        """
        해당 universe의 index 가격을 조회
        :param universe: kospi, kosdaq, kospi200, kosdaq150
        :param start_date:
        :param end_date:
        :return:
        """
        if universe in ('kospi', 'kosdaq'):
            universe = universe.upper()
        elif universe in ('kospi200', 'kosdaq150'):
            if universe == 'kospi200':
                universe = 'KOSPI_200'
            elif universe == 'kosdaq150':
                universe = 'KOSDAQ_150'

        data_index_candle = Table('data_indexcandleday')
        query = MySQLQuery.from_(data_index_candle).select('*').where(Criterion.all([
            data_index_candle.ticker == universe,
            data_index_candle.date >= start_date,
            data_index_candle.date <= end_date
        ]))
        df = self.executor.sql(query.get_sql())
        df = df.drop(['id', 'ticker'], axis=1)

        return df

    def get_exchange_rate(self, exchange_index: ExchangeRateEnum, start_date: date, end_date: date) -> DataFrame:
        """
        달러, 유로, 엔 환율 조회
        :param exchange_index:
        :param start_date:
        :param end_date:
        :return:
        """
        data_exchange_rate = Table('data_exchangeratecandleday')
        query = MySQLQuery.from_(data_exchange_rate).select('*').where(Criterion.all([
            data_exchange_rate.ticker == exchange_index.value,
            data_exchange_rate.date >= start_date,
            data_exchange_rate.date <= end_date
        ]))
        df = self.executor.sql(query.get_sql())
        df = df.drop(['id', 'ticker'], axis=1)
        return df
