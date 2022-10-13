from pipelines.retraining_pipeline import retraining_pipeline
from steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data, FetchDataConfig, print_dataframe_info

from datetime import datetime


def run_retraining_pipeline():
    print("running pipeline")
    train_start_date = datetime(2022, 6, 1)
    train_end_date = datetime(2022, 6, 30)

    val_start_date = datetime(2022, 7, 1)
    val_end_date = datetime(2022, 7, 31)

    fetch_train_data_config = FetchDataConfig(start_date=str(train_start_date.date()), end_date=str(train_end_date.date()))
    fetch_val_data_config = FetchDataConfig(start_date=str(val_start_date.date()), end_date=str(val_end_date.date()))

    pipe = retraining_pipeline(
        fetch_train_data=fetch_train_data(config=fetch_train_data_config),
        fetch_val_data=fetch_val_data(config=fetch_val_data_config),
        print_dataframe_info=print_dataframe_info()
    )
    pipe.run()


if __name__ == "__main__":
    run_retraining_pipeline()