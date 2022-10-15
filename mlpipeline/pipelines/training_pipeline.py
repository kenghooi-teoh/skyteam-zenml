from zenml.pipelines import pipeline


# TODO
# - eventually write model to model registry with evaluation scores
@pipeline(enable_cache=False)
def training_pipeline(
        fetch_train_data,
        fetch_val_data,
        fetch_label_data,
        feature_engineer_train,
        feature_engineer_val,
        training_data_preparation,
        train_xgb_model,
        evaluate_model,
        deployment_trigger,
        model_deployer
):
    train_df = fetch_train_data()
    val_df = fetch_val_data()
    label = fetch_label_data()

    train_feat = feature_engineer_train(train_df)
    val_feat = feature_engineer_val(val_df)

    x_train, y_train, x_val, y_val = training_data_preparation(train_feat, val_feat, label)

    model = train_xgb_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    accuracy = evaluate_model(model, x_val, y_val)

    deployment_decision = deployment_trigger(accuracy)

    service = model_deployer(deployment_decision, model)