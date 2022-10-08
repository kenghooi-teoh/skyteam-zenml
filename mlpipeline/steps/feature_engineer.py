import pandas as pd


class FeatureEngineer:

    def feature_engineer(self, df):
        # FEATURE ENGINEERING FROM
        # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
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
        print('shape after engineering', df.shape)

        return df
