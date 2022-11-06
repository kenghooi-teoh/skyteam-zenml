# Default prediction with Zenml
### What we built 
A simple ML pipeline that helps to perform loan default prediction. The actual tasks carried out by the pipeline include: 
- dataset preparation
- feature engineering
- model training and retraining
- model serving

The submission also includes a GUI that allows users to:
1. run batch inference
![image](https://user-images.githubusercontent.com/22209561/200164239-b41567ad-6f87-48e8-821a-be2f050eb8aa.png)
3. view business reports (outcome of prediction)
![image](https://user-images.githubusercontent.com/22209561/200164320-969c169d-3700-4b73-aaa9-f1537a2f2ad2.png)
4. monitor for data drift
![image](https://user-images.githubusercontent.com/22209561/200164948-d2c973db-af2d-4f7d-b32d-da77c89b7107.png)

### Why we chose this for your project to work on?
Our goal of participation is to learn Zenml. We began by exploring some public datasets and found the (AMEX Default Prediction dataset)[https://www.kaggle.com/competitions/amex-default-prediction/data], which contains borrowers’ 13-month profile data as learning features and the occurrence of a default event as the learning target. We modeled a binary classifier based on this dataset.
![image](https://user-images.githubusercontent.com/22209561/200164356-d97a9f3d-a947-49cc-b142-829615bf4c37.png)

We decided that this use case would be interesting to work on as we have not seen a lot of MLOps use cases shared publicly from an industry like banking, that requires very strict compliance and high transparency (during auditing) in the areas of dataset preparation, model development, model evaluation (and interpretation) and more.

### How we used ZenML to build out our pipeline
The first pipeline we created is the model training pipeline. The pipeline contains the following steps:
1. create training config
2. fetch training data
3. fetch validation data
4. fetch labels
5. feature engineer training data
6. feature engineer validation data
7. training data preparation
8. train an XGBoost model
9. create a prediction service
10. evaluate model and compare models to find the winning model
11. deploy last best model (MLflow deployer)

We also attempted to build 2 slightly different inference pipelines to support 2 use cases:
1. A batch inference pipeline that answers "how many borrowers will default their payment next month?"
2.  single-customer inference pipeline to predict if a given customer id will fulfill their payment or not.
These pipelines contain the following steps:
1. fetch inference data
2. feature engineer the inference data
3. load prediction service
4. make prediction with deployed model
5. store predictions

### What stack components we used and how we structured our code
We use the MLflow component in this submission. Despite using Evidently to perform drift detection, it was not integrated using Zenml's component.
All the Zenml pipeline code are kept in the app folder, and we followed the common code structure used by Zenml users by writing all the pipelines in a 'pipelines' folder while the steps are kept in a 'steps' folder. Here's an example of how we created the training pipeline using the individual steps:
![image](https://user-images.githubusercontent.com/22209561/200165599-9ac26f08-9152-4ce1-89d2-7eda08242415.png)

### Any demo or visual artifacts to showcase what we built
Run the `launch.sh` script in our (projet)[https://github.com/kenghooi-teoh/skyteam-zenml] to use the GUI in your browser.
