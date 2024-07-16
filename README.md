# Medical-Insurance-Cost-Prediction
Description:
This project involves predicting the cost of medical insurance based on various factors such as age, gender, BMI, number of children, smoking habits, and region. The prediction model is built using a Linear Regression algorithm to provide accurate cost estimations.

# Installation:

To run this project locally, follow these steps:

1-Clone the repository:

git clone https://github.com/Not-Prabhpreet/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction

2-Create a Virtual Environment

python -m venv venv

3-Activate the virtual environment:

venv\Scripts\activate

4-Install the required dependencies:

pip install -r requirements.txt


# Data Analysis: 

The project includes an extensive data analysis section where various distributions and counts are visualized. 

Age Distribution:


![image](https://github.com/user-attachments/assets/2c7e24cf-bc3c-4fa7-a0aa-8ab27e4e37cb)


Sex Distribution: 


![image](https://github.com/user-attachments/assets/98d8d185-ff03-4401-98b6-8ebf0a04e5b0)

BMI Distribution:


![image](https://github.com/user-attachments/assets/e9ab3b39-15f4-42c0-92cf-c16c457f4f1f)



Children Count:


![image](https://github.com/user-attachments/assets/5338f127-7e8d-464a-8f99-7b31772e6fa1)


Smoker Count:


![image](https://github.com/user-attachments/assets/5c159691-0ebb-4ca7-9169-b38895915885)

# Data Pre-Processing

The dataset is pre-processed by encoding catagorial variables.


insurance_dataset.replace({'sex':{'male':0, 'female':1}}, inplace=True)

insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

insurance_dataset.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest': 3}}, inplace=True)

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
print(X)
print(Y)


# Model Training and Evaluation
A Linear Regression model is trained and evaluated on the dataset.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value (training data):', r2_train)


test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value (test data):', r2_test)

# Results
The Linear Regression model provides the following R-squared values:

Training data: R squared value: 0.751
Test data: R squared value: 0.744

# Future Work
Deployment:
Currently working on a streamlit web app, I plan on dockerizing it and hosting it on AWS or Heroku
# Contact:
If you have any questions feel free to contact me at ppsingh@mun.ca




