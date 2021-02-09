from pypika import MySQLQuery, Table, Tables, Criterion
from typing import Optional

from utils import Executor
from datetime import date
from pandas import DataFrame
from utils.parameter import Universe, ExchangeRate

TABLE_RAW_CANDLE_DAY = "data_candleday"
TABLE_EDITED_CANDLE_DAY = "data_editedcandleday"


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
        data_trading_day = Table("data_iskoreatradingday")
        query = (
            MySQLQuery.from_(data_trading_day)
            .select(data_trading_day.date)
            .where(
                Criterion.all(
                    [
                        data_trading_day.is_tradable == "1",
                        data_trading_day.date >= start_date,
                        data_trading_day.date <= end_date,
                    ]
                )
            )
        )

        df = self.executor.sql(query.get_sql())
        trading_day_list = df["date"].to_list()
        return trading_day_list

    def get_past_universe_stock_list(self, universe: Universe) -> list:
        """
        과거에 한번이라도 universe에 포함된 종목 리스트 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :return: universe에 포함되었던 종목 리스트 반환
        """
        if universe in (Universe.total, Universe.kospi, Universe.kosdaq):
            data_ticker = Table("data_ticker")
            if universe == Universe.total:
                query = MySQLQuery.from_(data_ticker).groupby(data_ticker.ticker).select("*")
            else:
                query = (
                    MySQLQuery.from_(data_ticker)
                    .groupby(data_ticker.id)
                    .having(data_ticker.market == universe.name)
                    .select("*")
                )
            df = self.executor.sql(query.get_sql())
            past_universe_stock_list = df["ticker"].to_list()

            return past_universe_stock_list

        elif universe in (Universe.top350, Universe.kospi200, Universe.kosdaq150):
            if universe == Universe.top350:
                kospi200_past_universe_stock_list = self.get_past_universe_stock_list(Universe.kospi200)
                kosdaq150_past_universe_stock_list = self.get_past_universe_stock_list(Universe.kosdaq150)
                return kospi200_past_universe_stock_list + kosdaq150_past_universe_stock_list
            elif universe in (Universe.kospi200, Universe.kosdaq150):
                data_universe = Table(f"data_{universe.name}")
                query = (
                    MySQLQuery.from_(data_universe).groupby(data_universe.ticker_id).select(data_universe.ticker_id)
                )
                df = self.executor.sql(query.get_sql())
                past_universe_stock_list = df["ticker_id"].to_list()
                return past_universe_stock_list

    def __get_day_price_data(
        self,
        universe: Universe,
        table_name: str,
        trading_share: bool,
        trading_trend: bool,
        start_date: date,
        end_date: date,
    ) -> dict:

        past_universe_stock_list = self.get_past_universe_stock_list(universe)
        past_universe_stock_list = tuple(past_universe_stock_list)
        data_day_price, data_ticker = Tables(table_name, "data_ticker")

        query = MySQLQuery.from_(data_day_price).join(data_ticker).on(data_day_price.ticker_id == data_ticker.id)
        if trading_share is True:
            data_share = Table("data_daytradinginfo")
            query = query.join(data_share).on(
                Criterion.all(
                    [data_day_price.ticker_id == data_share.ticker_id, data_day_price.date == data_share.date]
                )
            )
        if trading_trend is True:
            data_trend = Table("data_daytradingtrend")
            query = query.left_join(data_trend).on(
                Criterion.all(
                    [data_day_price.ticker_id == data_trend.ticker_id, data_day_price.date == data_trend.date]
                )
            )
        if trading_share is False and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker)
        if trading_share is True and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker, data_share.cap, data_share.shares_out)
        if trading_share is False and trading_trend is True:
            query = query.select(
                data_day_price.star,
                data_ticker.ticker,
                data_trend.p_buy_vol,
                data_trend.p_buy_tr_val,
                data_trend.o_buy_vol,
                data_trend.o_buy_tr_val,
                data_trend.f_buy_vol,
                data_trend.f_buy_tr_val,
                data_trend.pension_f_buy_vol,
                data_trend.pension_f_tr_val,
            )
        if trading_share is True and trading_trend is True:
            query = query.select(
                data_day_price.star,
                data_ticker.ticker,
                data_share.cap,
                data_share.shares_out,
                data_trend.p_buy_vol,
                data_trend.p_buy_tr_val,
                data_trend.o_buy_vol,
                data_trend.o_buy_tr_val,
                data_trend.f_buy_vol,
                data_trend.f_buy_tr_val,
                data_trend.pension_f_buy_vol,
                data_trend.pension_f_tr_val,
            )
        query = query.where(
            Criterion.all(
                [
                    data_ticker.ticker.isin(past_universe_stock_list),
                    data_day_price.date >= start_date,
                    data_day_price.date <= end_date,
                ]
            )
        )
        df = self.executor.sql(query.get_sql())
        df = df.drop(["id"], axis=1)
        return df

    def get_day_price_data(
        self, universe: Universe, trading_share: bool, trading_trend: bool, start_date: date, end_date: date
    ) -> dict:

        """
        universe에 포함된 종목들의 무수정 주가 일봉 데이터 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :param trading_share: 상장주식수, 상장시가총액 데이터 포함여부
        :param trading_trend: 기간 순매수량, 외국인 순매수량 데이터 등 포함여부
        :param start_date:
        :param end_date:
        :return: 일봉 데이터
        """
        return self.__get_day_price_data(
            universe, TABLE_RAW_CANDLE_DAY, trading_share, trading_trend, start_date, end_date
        )

    def get_edited_day_price_data(self, universe: Universe) -> dict:
        """
        universe에 포함된 종목들의 수정주가 일봉 데이터 반환
        :param universe:
        :return: 일봉 데이터
        """
        return self.__get_day_price_data(universe, TABLE_EDITED_CANDLE_DAY)

    def __get_day_price_data_by_ticker(
        self, ticker, table_name, trading_share: bool, trading_trend: bool, start_date: date, end_date: date
    ):

        data_day_price, data_ticker = Tables(table_name, "data_ticker")

        query = MySQLQuery.from_(data_day_price).join(data_ticker).on(data_day_price.ticker_id == data_ticker.id)
        if trading_share is True:
            data_share = Table("data_daytradinginfo")
            query = query.join(data_share).on(
                Criterion.all(
                    [data_day_price.ticker_id == data_share.ticker_id, data_day_price.date == data_share.date]
                )
            )
        if trading_trend is True:
            data_trend = Table("data_daytradingtrend")
            query = query.left_join(data_trend).on(
                Criterion.all(
                    [data_day_price.ticker_id == data_trend.ticker_id, data_day_price.date == data_trend.date]
                )
            )
        if trading_share is False and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker)
        if trading_share is True and trading_trend is False:
            query = query.select(data_day_price.star, data_ticker.ticker, data_share.cap, data_share.shares_out)
        if trading_share is False and trading_trend is True:
            query = query.select(
                data_day_price.star,
                data_trend.p_buy_vol,
                data_trend.p_buy_tr_val,
                data_trend.o_buy_vol,
                data_trend.o_buy_tr_val,
                data_trend.f_buy_vol,
                data_trend.f_buy_tr_val,
                data_trend.pension_f_buy_vol,
                data_trend.pension_f_tr_val,
            )
        if trading_share is True and trading_trend is True:
            query = query.select(
                data_day_price.star,
                data_share.cap,
                data_share.shares_out,
                data_trend.p_buy_vol,
                data_trend.p_buy_tr_val,
                data_trend.o_buy_vol,
                data_trend.o_buy_tr_val,
                data_trend.f_buy_vol,
                data_trend.f_buy_tr_val,
                data_trend.pension_f_buy_vol,
                data_trend.pension_f_tr_val,
            )
        query = query.where(
            Criterion.all(
                [data_ticker.ticker == ticker, data_day_price.date >= start_date, data_day_price.date <= end_date]
            )
        )
        df = self.executor.sql(query.get_sql())
        df = df.drop(["id"], axis=1)
        return df

    def get_day_price_data_by_ticker(
        self, ticker, trading_share: bool, trading_trend: bool, start_date: date, end_date: date
    ):
        """
        universe에 포함된 종목들의 수정 주가 일봉 데이터 반환
        :param ticker: 개별 종목의 ticker
        :param trading_share:
        :param trading_trend:
        :param start_date:
        :param end_date:
        :return: 해당 ticker의 일봉 데이터
        """
        return self.__get_day_price_data_by_ticker(
            ticker, TABLE_RAW_CANDLE_DAY, trading_share, trading_trend, start_date, end_date
        )

    def get_edited_day_price_data(self, ticker):
        """
        universe에 포함된 종목들의 수정 주가 일봉 데이터 반환
        :param ticker: 개별 종목의 ticker
        :return: 해당 ticker의 일봉 데이터
        """
        return self.__get_edited_day_price_data_by_ticker(ticker, TABLE_EDITED_CANDLE_DAY)

    def get_today_universe_stock_list(self, universe: Universe, today: date) -> list:
        """
        당일의 universe에 포함된 종목 리스트 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :param today: 당일 날짜
        :return: 해당 universe의 종목 리스트
        """
        if universe in (Universe.kospi, Universe.kosdaq, Universe.kospi200, Universe.kosdaq150):
            data_universe = Table(f"data_{universe.name}history")
            query = MySQLQuery.from_(data_universe).select(data_universe.tickers).where(data_universe.date == today)
            df = self.executor.sql(query.get_sql())
            today_universe_stock_list = df["tickers"].iloc[0].split(",")
        elif universe in (Universe.total, Universe.top350):
            if universe == Universe.total:
                today_universe_stock_list = self.get_today_universe_stock_list(
                    Universe.kospi, today
                ) + self.get_today_universe_stock_list(Universe.kosdaq, today)
            elif universe == Universe.top350:
                today_universe_stock_list = self.get_today_universe_stock_list(
                    Universe.kospi200, today
                ) + self.get_today_universe_stock_list(Universe.kosdaq150, today)

        return today_universe_stock_list

    def get_index_day_price_data(self, universe: Universe, start_date: date, end_date: date) -> Optional[DataFrame]:
        """
        해당 universe의 index 가격을 조회
        :param universe: kospi, kosdaq, kospi200, kosdaq150
        :param start_date:
        :param end_date:
        :return:
        """
        if universe in (Universe.kospi, Universe.kosdaq):
            ticker = universe.name.upper()
        elif universe in (Universe.kospi200, Universe.kosdaq150):
            if universe == Universe.kospi200:
                ticker = "KOSPI200"
            elif universe == Universe.kosdaq150:
                ticker = "KOSDAQ150"

        data_index_candle = Table("data_indexcandleday")
        query = (
            MySQLQuery.from_(data_index_candle)
            .select("*")
            .where(
                Criterion.all(
                    [
                        data_index_candle.ticker == ticker,
                        data_index_candle.date >= start_date,
                        data_index_candle.date <= end_date,
                    ]
                )
            )
        )
        df = self.executor.sql(query.get_sql())
        df = df.drop(["id", "ticker"], axis=1)

        return df

    def get_exchange_rate(self, exchange_index: ExchangeRate, start_date: date, end_date: date) -> DataFrame:
        """
        달러, 유로, 엔 환율 조회
        :param exchange_index:
        :param start_date:
        :param end_date:
        :return:
        """
        data_exchange_rate = Table("data_exchangeratecandleday")
        query = (
            MySQLQuery.from_(data_exchange_rate)
            .select("*")
            .where(
                Criterion.all(
                    [
                        data_exchange_rate.ticker == exchange_index.value,
                        data_exchange_rate.date >= start_date,
                        data_exchange_rate.date <= end_date,
                    ]
                )
            )
        )
        df = self.executor.sql(query.get_sql())
        df = df.drop(["id", "ticker"], axis=1)
        return df
