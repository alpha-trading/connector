from enum import Enum, IntEnum


class Market(Enum):
    kospi = 1
    kosdaq = 2
    etf = 3


class Universe(IntEnum):
    total = -1
    kospi = 0
    kosdaq = 1


class ExchangeRate(Enum):
    dollar = "USDKRW"
    euro = "EURKRW"
    yen = "JPYKRW"


class MovingAverage(Enum):
    sma = 1
    ema = 2
    ewma = 3
    wma = 4


class KindOfAverage(Enum):
    average = 1
    min = 2
    max = 3
    first = 4
    dense = 5


class UnitPeriod(Enum):
    month = 1
    year = 2


class Table(Enum):
    candle_day = "data_candleday"
    ticker = "data_ticker"
    ticker_history = "data_tickerhistory"
    edited_candle_day = "data_editedcandleday"
    day_trading_trend = "data_daytradingtrend"
    day_trading_info = "data_daytradinginfo"
    is_korea_trading_day = "data_iskoreatradingday"
    exchange_rate_candle_day = "data_exchangeratecandleday"


class Field(Enum):
    ticker_id = ("ticker_id", Table.candle_day)
    open = ("open", Table.candle_day)
    low = ("low", Table.candle_day)
    high = ("high", Table.candle_day)
    close = ("close", Table.candle_day)
    vol = ("vol", Table.candle_day)
    tr_val = ("tr_val", Table.candle_day)
    p_buy_vol = ("p_buy_vol", Table.day_trading_trend)
    org_buy_vol = ("org_buy_vol", Table.day_trading_trend)
    f_buy_vol = ("f_buy_vol", Table.day_trading_trend)
    pen_buy_vol = ("pen_buy_vol", Table.day_trading_trend)
    f_buy_tr_val = ("f_buy_tr_val", Table.day_trading_trend)
    org_buy_tr_val = ("org_buy_tr_val", Table.day_trading_trend)
    p_buy_tr_val = ("p_buy_tr_val", Table.day_trading_trend)
    pen_buy_tr_val = ("pen_buy_tr_val", Table.day_trading_trend)
    etc_buy_tr_val = ("etc_buy_tr_val", Table.day_trading_trend)
    etc_buy_vol = ("etc_buy_vol", Table.day_trading_trend)
    cap = ("cap", Table.day_trading_info)
    shares_out = ("shares_out", Table.day_trading_info)

    def __init__(self, column_name: str, table: Table):
        self.column_name = column_name
        self.table = table

    def get_full_name(self):
        return f"{self.table.name}.{self.column_name}"
