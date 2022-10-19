from zenml.pipelines import pipeline


# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False)
def training_pipeline(
        training_config,
        fetch_train_data,
        fetch_val_data,
        fetch_label_data,
        feature_engineer_train,
        feature_engineer_val,
        training_data_preparation,
        train_xgb_model,
        prediction_service_loader,
        evaluate_model,
        model_deployer
):
    is_retraining = training_config()

    train_df = fetch_train_data()
    val_df = fetch_val_data()
    label = fetch_label_data()

    train_feat = feature_engineer_train(train_df)
    val_feat = feature_engineer_val(val_df)

    x_train, y_train, x_val, y_val = training_data_preparation(train_feat, val_feat, label)

    model = train_xgb_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    service = prediction_service_loader()

    deployment_decision = evaluate_model(model, service, x_val, y_val, is_retraining)

    model_deployer(deployment_decision, model)