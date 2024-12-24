import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    """
    为数据集添加时间特征: year、month、quarter、day
    :param df: DataFrame
    :return: DataFrame
    """

    # 添加新特征
    df['year'] = df['date'].dt.year       # 提取年份
    df['month'] = df['date'].dt.month     # 提取月份
    df['quarter'] = df['date'].dt.quarter # 提取季度

    base_date = pd.Timestamp('2002-01-01')
    df['day'] =  (df['date'].dt.tz_localize(None) - base_date).dt.days         # 提取日期
    
    return df


def split_data(data, cutoff=3):
    """
    划分数据集
    :param data: stock's
    :param months: 2 or 3

    :return: (data_train, data_test), (VZ_train, VZ_test), (SBUX_train, SBUX_test)
    """
    train = data[(data["year"] < 2011) |  \
                ((data["year"] == 2011) & (data["quarter"] < cutoff))]
    test = data[(data["year"] == 2011) & (data["quarter"] >= cutoff)]

    return train, test

