from pipelines.batch_inference_pipeline import batch_inference_pipeline
from steps.data_fetcher import fetch_ondemand_inference_data
from steps.feature_engineer import feature_engineer_inference_batch
from steps.predictor import predictor


def run_batch_inference_pipeline():
    print("running pipeline")

    pipe = batch_inference_pipeline(
        inference_data_fetcher=fetch_ondemand_inference_data(),
        feature_enginee=feature_engineer_inference_batch(),
        predictor=predictor()
    )
    pipe.run()


if __name__ == "__main__":
    run_batch_inference_pipeline()
