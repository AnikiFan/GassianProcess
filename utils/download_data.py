from tiingo import TiingoClient
import pandas as pd
import numpy as np

def download_data(Key):
    # 对于每个 API, 每支股票的历史数据短时间内(3 min 左右)不能重复下载，否则权限报错。
    # 除了 tiingo. 其他平台的 API 也可以使用。但是需要付费。而且大部分平台不直接提供 adjusted close price.

    # 配置 API Key
    config = {"api_key": Key}
    client = TiingoClient(config)

    # 获取 VZ 的历史数据
    ticker = ["VZ", "HPQ", "SBUX"]
    data = {
        t: client.get_dataframe(
            t,
            metric_name="adjClose",
            startDate="2002-01-01",
            endDate="2011-12-31",
        )
        for t in ticker
    }
    for t, d in data.items():
        d = pd.DataFrame(d)
        d.to_csv(f"data/{t}.csv", index=True)

if __name__ == "__main__":
     # hopeymir's tiingo API Key(has changed since this project). Please use your own API Key.
    download_data("1e1e81db34d6b1717cfba35f4c7a568f36c5d693")  
