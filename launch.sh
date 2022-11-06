#!/bin/bash

echo "Starting mysql container"
docker run --rm -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=zenml -d mysql
sleep 10

echo "Populating db with data"
python data-ingestion/load_data.py

cd app
echo "Starting streamlit"
streamlit run Home.py

