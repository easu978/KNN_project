# KNN_Project_Data

## Table of Contents
- [KNN\_Project\_Data](#knn_project_data)
  - [Table of Contents](#table-of-contents)
    - [Project Overview](#project-overview)
    - [Data Sources](#data-sources)
    - [Tools](#tools)
    - [Data Cleaning/Preparation](#data-cleaningpreparation)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Data Analysis](#data-analysis)
    - [Train Test Split](#train-test-split)
    - [Model Training](#model-training)
    - [Predictions](#predictions)
    - [Model Evaluation](#model-evaluation)
    - [Results/Findings](#resultsfindings)
    - [Recommendations](#recommendations)
    - [Limitations](#limitations)
    - [References](#references)
### Project Overview
This project is aimed at harnessing the powers of K Nearest Neighbors (KNN) algorithm to predict the Target Class based off of a data set whose features are classified.

![pairplot1](https://github.com/easu978/KNN_project/assets/151114298/a476ad01-9fd1-4578-8a17-3be0f1b0dd49)
![lineplot](https://github.com/easu978/KNN_project/assets/151114298/79449e83-d729-4ef7-8a3e-02108c027011)


### Data Sources
KNN_Project_Data: The data used for this analysis is the "KNN_Project_Data.csv" file, and it contains the information needed to carry out the classification task, without necessarily knowing what each feature stands for.
### Tools
- Pandas
- Numpy
- Matplotlib
- Seaborn
### Data Cleaning/Preparation
  In this phase, the following procedures were carried out:
- Data was read from source and inspected.
- Data was scaled using StandardScaler
### Exploratory Data Analysis
Since the features of the data set are classified, the data was explored using Seaborn’s pairplot, with hue based off the “Target Class”, so as to get insight on which features was closest to the Class.
### Data Analysis
```python
df = pd.read_csv('KNN_Project_Data')
sns.pairplot(data=df,hue='TARGET CLASS')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_features = pd.DataFrame(scaled_features,columns=df.columns[:-1])
```
### Train Test Split
In this phase, our data set is split into two categories:
- Training set : Here, our data is trained on all other features except the "TARGET CLASS"
- Test set : Our data is tested on the X_test, which will be compared to the y_test by our model(KNN)
```python
  from sklearn.model_selection import train_test_split
  X = df_feat
  y = df['TARGET CLASS']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```
### Model Training
The model(KNN) was trained on the X features (classified)
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
```
### Predictions
The model was used to predict the X_test
```python
predictions = knn.predict(X_test)
```
### Model Evaluation
The performance of the KNN algorithm was evaluated using the classification_report,confusion_matrix metrics;
```python
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))
```
### Results/Findings
With 72% precision (accuracy), the KNN algorithm predicted as follows:
- 109, 107 instances of correct prediction to TARGET CLASS 1,0
- 41, 43 instances of incorrect predictions to the TARGET CLASS
### Recommendations
In order to improve on the model's performance, the following can be done:
1. Aggregate the predictions, except for those that equal the y_test, into a list
2. call the list in (1) above, Error_rate
3. consider a range of values for K higher than 1(say,K=23,31), and re-train the model accordingly
4. plot a simple lineplot of Error_rate vs K-values(see plots attached).
```python
Error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    Error_rate.append(np.mean(pred_i!=y_test))
```
```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),Error_rate,color='blue',marker='o',markerfacecolor='red',linestyle='dashed',markersize=10)
plt.title('Error_rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error_rate')
```
### Limitations
Since the features of the data set are classified, further engineering on the features could not be carried out before deploying the KNN algorithm.
### References
1. [Udemy](https://Udemy.com)
2. [Stack Overflow](https://stack.com)


