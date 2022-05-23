# Logistic Regression to Predict the Final Grade of a Student


<img width="846" alt="Log-Reg-Graph" src="https://user-images.githubusercontent.com/91611463/169881626-c905da51-913b-43cd-9e41-3e5912c3256a.png">





## Purpose 

The algorithm interrogates whether the going out with friends, workday alcohol consumption, weekend alcohol consumption, going out with friends, and freetime after school is in correlation with grades. 




## Required Packages 

1. Pandas 
   for data manipulation and analysis
   
2. Matplotlib
    to plot the data
    
3. Sklearn
  to apply logistic regression
  
4. Seaborn
  to create the heat map
  
  

```python
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
data = pd.read_csv("student-mat.csv")
```
 


### Seaborn Heat Map


```python
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
```

<img width="675" alt="Screen Shot 2022-05-23 at 21 24 04" src="https://user-images.githubusercontent.com/91611463/169883058-595c9f90-b698-4068-be66-0b26cee160d4.png">


### Manipulating the Dataset


##### Creating a new dataset called "new_data" that only contains the required columns and displaying first 15 rows

```python
new_data = data.loc[:, ["Dalc", "Walc", "goout", "freetime", "G3"]]
new_data.head(15)
```

<img width="291" alt="Screen Shot 2022-05-23 at 21 26 15" src="https://user-images.githubusercontent.com/91611463/169883369-c40006c7-b852-4f7c-a847-4669af285bf8.png">




** Since the dataset did not contain any empty or duplicated rows or columns, I did not erase any. **


#### Taking a Peek at the dataset:


```python
new_data.head(15)
```

<img width="282" alt="Screen Shot 2022-05-23 at 21 39 39" src="https://user-images.githubusercontent.com/91611463/169885297-281a2da0-b7ab-4433-b16f-dd30f5f8a402.png">


#### Build your Model


1. Now I will select the variables I want to use in my regression model and split the training and test sets calling ```python train_test_split()```:


```python
from sklearn.model_selection import train_test_split

# Selected_features = ['Dalc','Walc','goout','freetime']

X = new_data.drop("G3",axis=1)
y = new_data.G3.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
```

2. Now I can train your model, by calling ```python fit()``` with my training data, and print out its result:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```


<img width="823" alt="Screen Shot 2022-05-23 at 21 49 07" src="https://user-images.githubusercontent.com/91611463/169886652-aae90cee-c18d-4733-b4af-62093441d30e.png">



#### Conclusion 

The correlation between the variables in the X-axis and grade were insignificant, with an accuracy score of 0.139. Therefore, I concluded that I can spend time with my friends, and spend my free time with a light heart.










