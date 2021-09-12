# supervised_ML_task01
#importing libraries
 
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
#reading data from webwebsite 

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
sample_data = pd.read_csv(url)
print("your imported data is here")

sample_data.head(15)
# Plotting the distribution of scores
sample_data.plot(x='Hours', y='Scores', style='*')  
plt.title(' Percentage vs Hours ')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
#preparing data
X = sample_data.iloc[:, :-1].values  
Y = sample_data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train,Y_test = train_test_split(X, Y,test_size=0.2, random_state=0) 
#training the algo.
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()
print(X_test) # Testing data - In Hours
Y_pred = regressor.predict(X_test) # Predicting the scores
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
print(df) 
sample_data.shape
sample_data.plot(kind='box')
#change the dataframe variable
Hours = pd.DataFrame(sample_data['Hours'])
Scores=pd.DataFrame(sample_data['Scores'])
#build linear regression model
from sklearn import linear_model
lm= linear_model.LinearRegression()
model = lm.fit(Hours,Scores)
model.coef_
model.intercept_
model.score(Hours,Scores)
#predict the new value of Score
Hours = ([9.25])
Hours = pd.DataFrame(Hours)
Y=model.predict(X)
Y = pd.DataFrame(Y)
df
#correlation coefficients 
sample_data.corr()  
# predict the score if a student study 9.25 H/D
Hours = 9.25
own_pred = regressor.predict([[Hours]])
print("No of Hours = {} ".format(Hours))
print("predicted score ={} ".format(own_pred))
print ("If a student study 9.25 H/D then they can gain 93.69% score as we predict")
from sklearn import metrics
# finding the value of  mean absolute error
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))
# finding the value of mean square error
print ('Mean Square Error:', metrics.mean_squared_error(Y_test,Y_pred))
