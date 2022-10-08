import pandas as pd
from zenml.steps import Output
from typing import Tuple

class DataPreprocessor:


    def _merge_data(self, preprocessed_train_df: pd.DataFrame, preprocessed_valid_df: pd.DataFrame, label_df: pd.DataFrame) ->\
        Tuple(pd.DataFrame, pd.DataFrame):
        """Merge preprocessed features after feature engineering

        Args:
            preprocessed_train_df (pd.DataFrame): feature engineered train features
            preprocessed_valid_df (pd.DataFrame): feature engineered valid features
            label_df (pd.DataFrame): raw label df
        Returns:
            (pd.DataFrame, pd.DataFrame): train feature label dataframe, validation feature label dataframe
        
        """
        train_fea_label = pd.merge(preprocessed_train_df, label_df, left_on="customer_ID", right_on="customer_ID", how="left")
        valid_fea_label = pd.merge(preprocessed_valid_df, label_df, left_on="customer_ID", right_on="customer_ID", how="left")
        return train_fea_label, valid_fea_label

    def _split_label_feature(self, df_fea_label: pd.DataFrame) -> Tuple(pd.DataFrame, pd.Series):
        """
        Split dataframe into feature and labels

        Args:
            df_fea_label (pd.DataFrame): DataFrame contains feature and label

        Returns:
            (pd.DataFrame, pd.DataFrame): feature dataframe, label dataframe
        """
        df_X = df_fea_label[df_fea_label.columns[1:-1]]
        df_y = df_fea_label[df_fea_label.columns[-1]]
        return df_X, df_y

    def clean_data(self, preprocessed_train_df: pd.DataFrame, preprocessed_valid_df: pd.DataFrame, label_df: pd.DataFrame) ->\
        Tuple(pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """
        Perform data cleaning by merging relevant label and feature based on customer ID

        Args:
            preprocessed_train_df (pd.DataFrame): feature engineered train features
            preprocessed_valid_df (pd.DataFrame): feature engineered valid features
            label_df (pd.DataFrame): raw label df

        Returns:
                (pd.DataFrame, pd.DataFrame): feature dataframe, label dataframe
        """
        train_fea_label, valid_fea_label = self._merge_data(preprocessed_train_df=preprocessed_train_df,
                                                            preprocessed_valid_df=preprocessed_valid_df, 
                                                            label_df=label_df)
        train_X, train_y = self._split_label_feature(train_fea_label)
        valid_X, valid_y = self._split_label_feature(valid_fea_label)
        return train_X, train_y, valid_X, valid_y
