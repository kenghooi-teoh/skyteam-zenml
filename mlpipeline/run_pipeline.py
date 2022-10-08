from mlpipeline.pipelines.training_pipeline import training_pipeline
from mlpipeline.steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data
from mlpipeline.steps.data_preprocessor import clean_data
# from mlpipeline.steps.trainer import train_xgb_model
from mlpipeline.steps.feature_engineer import feature_engineer


def run_training_pipeline():
    pipe = training_pipeline(
        fetch_train_data=fetch_train_data(),
        fetch_val_data=fetch_val_data(),
        fetch_label_data=fetch_label_data(),
        feature_engineer=feature_engineer(),
        clean_data=clean_data(),
        # train_xgb_model()
    )
    pipe.run()


if __name__ == "main":
    run_training_pipeline()
