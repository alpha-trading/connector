from datetime import datetime, date


def year_to_datetime(year: int):
    return datetime(year=year, month=1, day=1)


def get_hoga(price, market):
    """
    주가에 따른 1호가 가격을 구하는 함수
    :param price: 주가
    :param market: kospi / kosdaq
    :return: 1호가 가격
    """
    if market == 'kospi':
        if price >= 500000:
            hoga = 1000
        elif price >= 100000:
            hoga = 500
        elif price >= 50000:
            hoga = 100
        elif price >= 10000:
            hoga = 50
        elif price >= 5000:
            hoga = 10
        elif price >= 1000:
            hoga = 5
        else:
            hoga = 1

    elif market == 'kosdaq':
        if price >= 50000:
            hoga = 100
        elif price >= 10000:
            hoga = 50
        elif price >= 5000:
            hoga = 10
        elif price >= 1000:
            hoga = 5
        else:
            hoga = 1

    return hoga


def get_bid_price(price, market):
    """
    살 가격을 구하는 함수
    :param price: 현재가
    :param market: 시장
    :return: 매수 가격
    """
    hoga = get_hoga(price, market)
    return (price // hoga) * hoga + hoga


def get_ask_price(price, market):
    """
    팔 가격을 구하는 함수
    :param price: 현재가
    :param market: 시장
    :return: 매도 가격
    """
    hoga = get_hoga(price, market)
    if price % hoga == 0:
        return (price // hoga) * hoga - hoga
    else:
        return (price // hoga) * hoga


def get_upper_limit(today):
    if today < date(2015, 6, 15):
        return 1.15
    else:
        return 1.3


def get_upper_limit_price(yesterday_close, today, market):
    """
    당일 상한가를 반환하는 함수
    :param yesterday_close: 전일 종가
    :param today: 당일 날짜
    :param market: kospi 또는 kosdaq
    :return: 당일 상한가
    """
    upper_limit = get_upper_limit(today)
    predict_upper_limit_price = yesterday_close * upper_limit
    hoga = get_hoga(predict_upper_limit_price, market)
    real_upper_limit_price = (predict_upper_limit_price // hoga) * hoga

    return real_upper_limit_price
