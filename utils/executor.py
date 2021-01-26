# -*- coding:utf-8 -*-
import os
from urllib.parse import urlparse

import pymysql
import dotenv
import pandas as pd


def parse(url: str):
    result = urlparse(url)
    return {
        "host": result.hostname,
        "port": result.port if result.port else 3306,
        "user": result.username,
        "passwd": result.password,
        "db": result.path[1:] if result.path else "",
        "charset": "utf8",
    }


class Executor:
    def __init__(self, database_url: str = None):
        if not database_url:
            dotenv.load_dotenv()
            database_url = os.environ.get("DATABASE_URL")

        self.connector = pymysql.connect(**parse(database_url))

    def sql(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.connector)

    def __del__(self):
        self.connector.close()
