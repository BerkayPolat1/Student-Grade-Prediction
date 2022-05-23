# Logistic Regression to Predict the Final Grade of a Portuguese Student


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
 


### Seaborn Heat Map


```python
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
```

<img width="675" alt="Screen Shot 2022-05-23 at 21 24 04" src="https://user-images.githubusercontent.com/91611463/169883058-595c9f90-b698-4068-be66-0b26cee160d4.png">


### Manipulating the Dataset

```python
new_data = 	data.loc[:, ["Dalc", "Walc", "goout", "freetime", "G3"]]
new_data.head(15)
```

<img width="291" alt="Screen Shot 2022-05-23 at 21 26 15" src="https://user-images.githubusercontent.com/91611463/169883369-c40006c7-b852-4f7c-a847-4669af285bf8.png">




** Since the dataset did not contain any empty or duplicated rows or columns, I did not erase any. **


#### Taking a Peek at the dataset:


```python
new_data.head(15)
new_data.info
```














