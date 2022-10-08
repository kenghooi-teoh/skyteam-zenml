from materializer.custom_materializer import cs_materializer
from pipelines.training_pipeline import train_pipeline
from steps.data_preprocessor import clean_data
from steps.model_evaluator import evaluation
from steps.data_fetcher import ingest_data
from steps.trainer import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


def run_training():
    training = train_pipeline(
        ingest_data(),
        clean_data().with_return_materializers(cs_materializer),
        train_model(),
        evaluation(),
    )

    training.run()


if __name__ == "__main__":
    run_training()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
