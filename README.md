<b>Project Description</b>
Project contains a web app which can entry a real-life disater message and get result of that be devided into some of 36 pre-defined categories. Therefore we can use it to sent these message to appropriate disaster relief agency.

Project contains 3 part:
* ETL Pipeline which extract data, clean and save data into database.
* Machine Learning Pipeline which is a Natural Language Processing Model to classify text message into categories.
* Web App to show result and entry text to classify.

<b>File Description</b>
![image](https://user-images.githubusercontent.com/117885173/214674868-73b780e8-06f2-4683-856c-dfb7a578ddc3.png)
  
<b>Instructions</b>

1. To run ETL pipeline that extract, clean data and store data: #python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db
2. To run ML pipeline that trains classify model and save to file: #models/train_classifier.py data/Disaster_Response.db models/classifier.pkl

3. To run web app: #python run.py and then go to http://localhost:3000/

<b>Screenshots</b>
![image](https://user-images.githubusercontent.com/117885173/214674380-542d450b-821a-40db-82ca-728278415454.png)
![image](https://user-images.githubusercontent.com/117885173/214674605-eb3eda49-6073-4c9d-bb40-ea51b27e8838.png)
![image](https://user-images.githubusercontent.com/117885173/214674699-3fb49495-097b-45c6-8860-2352992498fa.png)


