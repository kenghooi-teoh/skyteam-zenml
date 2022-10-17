from datetime import datetime

from pipelines.batch_inference_pipeline import batch_inference_pipeline
from steps.data_fetcher import fetch_batch_inference_data, FetchDataConfig
from steps.feature_engineer import feature_engineer_inference_batch
from steps.prediction_service_loader import prediction_service_loader, PredictionServiceLoaderStepConfig
# from steps.prediction_storer import prediction_storer, StorePredictionConfig
from steps.predictor import predictor


def run_batch_inference_pipeline():
    print("running batch inference pipeline")
    val_start_date = datetime(2022, 7, 1)
    val_end_date = datetime(2022, 7, 31)

    fetch_val_data_config = FetchDataConfig(
        start_date=str(val_start_date.date()),
        end_date=str(val_end_date.date())
    )

    predictor_service_config = PredictionServiceLoaderStepConfig(
        pipeline_name="training_pipeline",
        step_name="model_deployer",
        model_name="xgboost"
    )

    pipe = batch_inference_pipeline(
        inference_data_fetcher=fetch_batch_inference_data(config=fetch_val_data_config),
        feature_engineer=feature_engineer_inference_batch(),
        prediction_service_loader=prediction_service_loader(config=predictor_service_config),
        predictor=predictor(),
        # prediction_storer=prediction_storer()
    )
    pipe.run()


if __name__ == "__main__":
    run_batch_inference_pipeline()
