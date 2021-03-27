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
    simulating = "data_simulating"
    exchange_rate = "data_exchangeratetable"
    ticker_universe = "data_ticker_universe"


class Field(Enum):
    ticker_id = "ticker_id"
    date = "date"
    open = "open"
    low = "low"
    high = "high"
    close = "close"
    vol = "vol"
    tr_val = "tr_val"
    p_buy_vol = "p_buy_vol"
    org_buy_vol = "org_buy_vol"
    f_buy_vol = "f_buy_vol"
    pen_buy_vol = "pen_buy_vol"
    f_buy_tr_val = "f_buy_tr_val"
    org_buy_tr_val = "org_buy_tr_val"
    p_buy_tr_val = "p_buy_tr_val"
    pen_buy_tr_val = "pen_buy_tr_val"
    etc_buy_tr_val = "etc_buy_tr_val"
    etc_buy_vol = "etc_buy_vol"
    cap = "cap"
    shares_out = "shares_out"
    is_active = "is_active"
