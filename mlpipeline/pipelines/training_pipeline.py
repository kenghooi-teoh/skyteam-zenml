from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline


# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False, required_integrations=[MLFLOW])
def training_pipeline(
    fetch_train_data, fetch_val_data, fetch_label_data, feature_engineer_train, feature_engineer_val, clean_data,
        train_xgb_model
):
    """
    Args:

    Returns:s
    """
    ...

    train_df = fetch_train_data()
    val_df = fetch_val_data()
    label = fetch_label_data()
    print("data loaded")

    train_feat = feature_engineer_train(train_df)
    val_feat = feature_engineer_val(val_df)

    x_train, y_train, x_val, y_val = clean_data(train_feat, val_feat, label)
    print("data preprocessed")

    model = train_xgb_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    print("model trained")
