from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ENGINE = create_engine('mysql+pymysql://root:root@127.0.0.1:3306/zenml', echo=False)


def to_date_string(dt: datetime):
    return dt.strftime("%Y-%m-%d")


def amex_metric_mod(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def raw_pred_to_class(pred: Union[np.array, list]):
    """
    list of raw pred to int class
    :param pred: list or np.array
    :return: list
    """
    return list(map(lambda x: int(x >= 0.5), pred))


def load_df_to_sql(data: pd.DataFrame, table_name, connection, if_exists):  # TODO
    data.to_sql(table_name, con=connection, if_exists=if_exists, index=False)


def run_id_to_datetime(run_id):
    """
    'batch_inference_pipeline-17_Oct_22-23_57_48_920449' --> datetime.datetime(2022, 10, 17, 23, 57, 48, 920449)
    :param run_id:
    :return:
    """
    return datetime.strptime(run_id, '%d_%b_%y-%H_%M_%S_%f')

def _feature_engineer(df: pd.DataFrame):
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features]
    valid_cat_features = [fea for fea in cat_features if fea in all_cols]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = pd.DataFrame()
    if len(valid_cat_features) != 0:
        test_cat_agg = df.groupby("customer_ID")[valid_cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    df = pd.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    return df
