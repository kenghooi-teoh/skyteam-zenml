from zenml.pipelines import pipeline

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
        prediction_service_loader,
        service_model_evaluator,
        retraining_deployment_trigger,
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

    model_deployment_service = prediction_service_loader()
    accuracy_current = service_model_evaluator(model_deployment_service, x_val, y_val)

    deployment_decision = retraining_deployment_trigger(accuracy_current, accuracy)

    model_deployer(deployment_decision, model)