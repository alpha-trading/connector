from pypika import MySQLQuery, Table, Tables, Criterion, Field
from typing import Optional

from utils import Executor
import datetime
from pandas import DataFrame
from utils.parameter import Universe, ExchangeRate
from utils.parameter import Field as PhysicalField
from utils.parameter import Table as PhysicalTable


class Reader:
    def __init__(self, executor: Executor):
        self.executor = executor
        self.executor.set_read_mode(True)

    def get_simulating_data(
        self,
        universe_list: list,
        field_list: list,
        start_date: datetime.date,
        end_date: datetime.date,
        include_ticker_id_column=True,
        include_date_column=True,
    ):
        '''
        시뮬레이팅 및 전략 개발 기본 데이터 로딩
        :param universe_list: 불러올 universe 목록(utils.parameter.Universe[])
        :param field_list: 불러올 db field 목록(utils.parameter.Field[])
        :param start_date: 불러올 기간 시작일(datetime.date)
        :param end_date: 불러올 기간 종료일(datetime.date)
        :param include_ticker_id_column: ticker_id 컬럼을 포함하여 로딩할지 결정
        :param include_date_column: date 컬럼을 포함하여 로딩할지 결정
        :return:
        '''
        if field_list is None:
            field_list = [PhysicalField.__members__[x] for x in list(PhysicalField.__members__)]

        table = Table(PhysicalTable.simulating.value)
        exchange_rate_table = Table(PhysicalTable.exchange_rate.value)
        ticker_universe_table = Table(PhysicalTable.ticker_universe.value)

        query = (
            MySQLQuery.from_(table)
            .join(ticker_universe_table).using(PhysicalField.ticker_id.value, PhysicalField.date.value)     # universe
            .where(
                Criterion.all(
                    [
                        table.date >= start_date,
                        table.date <= end_date,
                        ticker_universe_table.universe in universe_list
                    ]
                )
            )
        )

        # 환율 정보
        for column in ExchangeRate:
            query.join(exchange_rate_table).on(
                Criterion.all(
                    [exchange_rate_table.date == table.date, exchange_rate_table.ticker == column]
                )
            )

        if include_ticker_id_column and PhysicalField.ticker_id not in field_list:
            field_list.append(PhysicalField.ticker_id)

        if include_date_column and PhysicalField.ticker_id not in field_list:
            field_list.append(PhysicalField.date)

        query = query.select(*[Field(x.column_name, table=Table(x.table.value)) for x in field_list])
        df = self.executor.sql(query.get_sql())
        return df
