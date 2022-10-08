from pipelines.training_pipeline import training_pipeline
from steps.data_fetcher import fetch_train_data, fetch_val_data, fetch_label_data
from steps.data_preprocessor import clean_data
from mlpipeline.steps.trainer import train_xgb_model
from steps.feature_engineer import feature_engineer_train, feature_engineer_val


def run_training_pipeline():
    print("running pipeline")
    pipe = training_pipeline(
        fetch_train_data=fetch_train_data(),
        fetch_val_data=fetch_val_data(),
        fetch_label_data=fetch_label_data(),
        feature_engineer_train=feature_engineer_train(),
        feature_engineer_val=feature_engineer_val(),
        clean_data=clean_data(),
        train_xgb_model=train_xgb_model()
    )
    pipe.run()


if __name__ == "__main__":
    run_training_pipeline()
