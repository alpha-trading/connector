from pypika import MySQLQuery, Table, Tables, Criterion, Field
from typing import Optional

from utils import Executor
import datetime
from pandas import DataFrame
from utils.parameter import Universe, ExchangeRate
from utils.parameter import Table as PhysicalTable
from utils.parameter import Field as PhysicalField


class Reader:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.executor.set_read_mode(True)

    def get_trading_day_list(self, start_date: datetime.datetime.date, end_date: datetime.datetime.date) -> list:
        """
        과거 거래일 리스트를 조회하는 함수
        :param start_date: back test 시작일
        :param end_date: back test 종료일
        :return: 과거 거래일 리스트 반환
        """
        data_trading_day = Table(PhysicalTable.is_korea_trading_day.value)
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

    def get_simulating_data(
        self,
        universe: Universe,
        field_list: list,
        start_date: datetime.date,
        end_date: datetime.date,
        append_ticker_id_column=True,
    ):
        ticker_history_table = Table(PhysicalTable.ticker_history.value)
        candle_table = Table(PhysicalTable.candle_day.value)
        trading_trend_table = Table(PhysicalTable.day_trading_trend.value)
        trading_info_table = Table(PhysicalTable.day_trading_info.value)
        exchange_rate_table = Table(PhysicalTable.exchange_rate_candle_day.value)

        query = (
            MySQLQuery.from_(ticker_history_table)
            .select("date")
            .where(
                Criterion.all(
                    [
                        ticker_history_table.date >= start_date,
                        ticker_history_table.date <= end_date,
                        ticker_history_table.is_active,
                    ]
                )
            )
        )

        if universe != Universe.total:
            query = query.where(Criterion.all([ticker_history_table.universe == universe]))

        # candle_day 를 요구
        if any([x.table == PhysicalTable.candle_day for x in field_list]):
            query = query.join(candle_table).using("date", "ticker_id")

        # trading_trend 를 요구
        if any([x.table == PhysicalTable.day_trading_trend for x in field_list]):
            query = query.join(trading_trend_table).using("date", "ticker_id")

        # trading_info 를 요구
        if any([x.table == PhysicalTable.day_trading_info for x in field_list]):
            query = query.join(trading_info_table).using("date", "ticker_id")

        # 환율 정보
        for column in ExchangeRate:
            query.join(exchange_rate_table).on(
                Criterion.all(
                    [exchange_rate_table.date == ticker_history_table.date, exchange_rate_table.ticker == column]
                )
            )

        if append_ticker_id_column and PhysicalField.ticker_id not in field_list:
            field_list.append(PhysicalField.ticker_id)

        query = query.select(*[Field(x.column_name, table=Table(x.table.value)) for x in field_list])
        df = self.executor.sql(query.get_sql())
        return df

    def get_past_universe_stock_list(self, universe: Universe) -> list:
        """
        과거에 한번이라도 universe에 포함된 종목 리스트 반환
        :param universe: total, kospi, kosdaq
        :return: universe에 포함되었던 종목 리스트 반환
        """
        data_ticker = Table(PhysicalTable.ticker.value)

        query = MySQLQuery.from_(data_ticker).select("ticker_id").distinct()
        if universe != Universe.total:
            query = query.where(data_ticker.universe == universe)
        df = self.executor.sql(query.get_sql())
        past_universe_stock_list = df["ticker"].to_list()

        return past_universe_stock_list

    def __get_day_price_data(
        self,
        universe: Universe,
        table_name: str,
        trading_share: bool,
        trading_trend: bool,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> dict:

        past_universe_stock_list = self.get_past_universe_stock_list(universe)
        past_universe_stock_list = tuple(past_universe_stock_list)
        data_day_price, data_ticker = Tables(table_name, "data_ticker")

        query = MySQLQuery.from_(data_day_price).join(data_ticker).on(data_day_price.ticker_id == data_ticker.id)
        if trading_share is True:
            data_share = Table(PhysicalTable.day_trading_info.value)
            query = query.join(data_share).on(
                Criterion.all(
                    [data_day_price.ticker_id == data_share.ticker_id, data_day_price.date == data_share.date]
                )
            )
        if trading_trend is True:
            data_trend = Table(PhysicalTable.day_trading_trend.value)
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
        self,
        universe: Universe,
        trading_share: bool,
        trading_trend: bool,
        start_date: datetime.date,
        end_date: datetime.date,
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
            universe, PhysicalTable.candle_day.value, trading_share, trading_trend, start_date, end_date
        )

    def get_edited_day_price_data(self, universe: Universe) -> dict:
        """
        universe에 포함된 종목들의 수정주가 일봉 데이터 반환
        :param universe:
        :return: 일봉 데이터
        """
        return self.__get_day_price_data(universe, PhysicalTable.edited_candle_day)

    def __get_day_price_data_by_ticker(
        self,
        ticker,
        table_name,
        trading_share: bool,
        trading_trend: bool,
        start_date: datetime.date,
        end_date: datetime.date,
    ):

        data_day_price, data_ticker = Tables(table_name, "data_ticker")

        query = MySQLQuery.from_(data_day_price).join(data_ticker).on(data_day_price.ticker_id == data_ticker.id)
        if trading_share is True:
            data_share = Table(PhysicalTable.day_trading_info.value)
            query = query.join(data_share).on(
                Criterion.all(
                    [data_day_price.ticker_id == data_share.ticker_id, data_day_price.date == data_share.date]
                )
            )
        if trading_trend is True:
            data_trend = Table(PhysicalTable.day_trading_trend.value)
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
        self, ticker, trading_share: bool, trading_trend: bool, start_date: datetime.date, end_date: datetime.date
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
            ticker, PhysicalTable.candle_day, trading_share, trading_trend, start_date, end_date
        )

    def get_edited_day_price_data(self, ticker):
        """
        universe에 포함된 종목들의 수정 주가 일봉 데이터 반환
        :param ticker: 개별 종목의 ticker
        :return: 해당 ticker의 일봉 데이터
        """
        return self.__get_edited_day_price_data_by_ticker(ticker, PhysicalTable.edited_candle_day)

    def get_today_universe_stock_list(self, universe: Universe, today: datetime.date) -> list:
        """
        당일의 universe에 포함된 종목 리스트 반환
        :param universe: kospi, kosdaq, kospi200, kosdaq150, top2000, top350(kospi200과 kosdaq150)
        :param today: 당일 날짜
        :return: 해당 universe의 종목 리스트
        """
        if universe in (Universe.kospi, Universe.kosdaq):
            data_universe = Table(f"data_{universe.name}history")
            query = MySQLQuery.from_(data_universe).select(data_universe.tickers).where(data_universe.date == today)
            df = self.executor.sql(query.get_sql())
            today_universe_stock_list = df["tickers"].iloc[0].split(",")
        else:
            today_universe_stock_list = self.get_today_universe_stock_list(
                Universe.kospi, today
            ) + self.get_today_universe_stock_list(Universe.kosdaq, today)

        return today_universe_stock_list

    def get_index_day_price_data(
        self, universe: Universe, start_date: datetime.date, end_date: datetime.date
    ) -> Optional[DataFrame]:
        """
        해당 universe의 index 가격을 조회
        :param universe: kospi, kosdaq
        :param start_date:
        :param end_date:
        :return:
        """
        if universe in (Universe.kospi, Universe.kosdaq):
            ticker = universe.name.upper()

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

    def get_exchange_rate(
        self, exchange_index: ExchangeRate, start_date: datetime.date, end_date: datetime.date
    ) -> DataFrame:
        """
        달러, 유로, 엔 환율 조회
        :param exchange_index:
        :param start_date:
        :param end_date:
        :return:
        """
        data_exchange_rate = Table(PhysicalTable.exchange_rate_candle_day.value)
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
