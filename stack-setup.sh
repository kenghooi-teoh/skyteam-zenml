zenml integration install mlflow sklearn xgboost -y

zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml stack update default -e mlflow_trackers