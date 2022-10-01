# Predicting loan default

**Problem statement**: TODO



## :snake: Python Requirements

```bash
pip install -r requirements.txt
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
``` 
The zenfile can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml stack set default
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack update default -d mlflow -e mlflow_tracker
```



### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

### Deployment Pipeline

We have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:


- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).


In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

To round it off, we deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit code:

```python
service = load_last_service_from_step(
    pipeline_name="continuous_deployment_pipeline",
    step_name="model_deployer",
    running=True,
)
...
service.predict(...)  # Predict on incoming data from the application
```

While this ZenFile trains and deploys a model locally, other ZenML integrations such as the [Seldon](https://github.com/zenml-io/zenml/tree/main/examples/) deployer can also be used in a similarly manner to deploy the model in a more production setting (such as on a Kubernetes cluster). We use MLflow here for the convenience of its local deployment.

![training_and_deployment_pipeline](_assets/training_and_deployment_pipeline_updated.png)

## :notebook: Diving into the code

You can run two pipelines as follows:

-  Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:


```bash
python run_deployment.py
```


## 🕹 Demo Streamlit App

There is a live demo of this project using [Streamlit](https://streamlit.io/) which you can find [here](https://share.streamlit.io/ayush714/customer-satisfaction/main). It takes some input features for the product and predicts the customer satisfaction rate using the latest trained models. If you want to run this Streamlit app in your local system, you can run the following command:-

```bash
streamlit run streamlit_app.py
```

## :question: FAQ

1. When running the continuous deployment pipeline, I get an error stating: `No Step found for the name mlflow_deployer`.

Solution: It happens because your artifact store is overridden after running the continuous deployment pipeline. So, you need to delete the artifact store and rerun the pipeline. You can get the location of the artifact store by running the following command:

```bash
zenml artifact-store describe
```

and then you can delete the artifact store with the following command:

**Note**: This is a dangerous / destructive command! Please enter your path carefully, otherwise it may delete other folders from your computer.

```bash
rm -rf PATH
```

2. When running the continuous deployment pipeline, I get the following error: `No Environment component with name mlflow is currently registered.`

Solution: You forgot to install the MLflow integration in your ZenML environment. So, you need to install the MLflow integration by running the following command:

```bash
zenml integration install mlflow -y
```
