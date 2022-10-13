from zenml.pipelines import pipeline
from datetime import datetime

# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False)
def retraining_pipeline(
        fetch_train_data,
        fetch_val_data,
        fetch_label_data,
        feature_engineer_train,
        feature_engineer_val,
        training_data_preparation,
        train_xgb_model,
        evaluate_model,

):
    train_df = fetch_train_data()
    val_df = fetch_val_data()



