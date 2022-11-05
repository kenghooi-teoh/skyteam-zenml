# Predicting loan default

**Problem statement**: TODO

## Python Requirements

```bash
pip install -r requirements.txt
```

## Initialize .zen environment

```bash
zenml init
```

## Setup stack

```bash
bash stack-setup.sh
```

## Populate mysql database

```bash
docker run --rm -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=zenml -d mysql
```
```bash
python data-ingestion/load_data.py
```

## Start Streamlit app

In `app` folder, run
```bash
streamlit run Home.py
```



