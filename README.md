# Connector

pymysql wrapper for execute sql with pandas dataframe
## Installation
```shell script
# In your shell
pip install -U git+https://github.com/alpha-trading/connector.git
# In colab
!pip install -U git+https://github.com/alpha-trading/connector.git
```

## Usage
```python
from connector import Executor

executor = Executor("mysql://user:password@db.url.com:3306/dbname")
# or use env var (env name: DATABASE_URL)
data_frame = executor.sql("SELECT * FROM your_table;")
print(data_frame)
```