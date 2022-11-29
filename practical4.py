# Linear Regression:

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
def coef_estimation(x, y):
    
    n = np.size(x)
    
    m_x, m_y = np.mean(x), np.mean(y)
    
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    return(b_0, b_1)

def plot_regression_line(x, y, b):
    
    plt.scatter(x, y, color = "m", marker = "o", s = 30)
    
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
        
def main():
   x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
   y = np.array([100, 300, 350, 500, 750, 800, 850, 900, 1050, 1250])
   b = coef_estimation(x, y)
   print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))
   plot_regression_line(x, y, b)
   
if __name__ == "__main__":
    main()        
    
# Output

Estimated coefficients:
b_0 = 154.5454545454545 
b_1 = 117.87878787878788

======================================================================================================

# Random Forest:

from sklearn import datasets   

iris = datasets.load_iris() 

print(iris.target_names) 

print(iris.feature_names) 

X, y = datasets.load_iris( return_X_y = True) 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70) 

from sklearn.ensemble import RandomForestClassifier 

import pandas as pd 

data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1], 

                     'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3], 

                     'species': iris.target}) 

print(data.head()) 

clf = RandomForestClassifier(n_estimators = 100)   

clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test) 

from sklearn import metrics   

print() 

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 
clf.predict([[3, 3, 2, 2]]) 
from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators = 100) 
clf.fit(X_train, y_train) 
import pandas as pd 
feature_imp = pd.Series(clf.feature_importances_, index = iris.feature_names).sort_values(ascending = False) 

feature_imp

# Output

['setosa' 'versicolor' 'virginica']
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
   sepallength  sepalwidth  petallength  petalwidth  species
0          5.1         3.5          1.4         0.2        0
1          4.9         3.0          1.4         0.2        0
2          4.7         3.2          1.3         0.2        0
3          4.6         3.1          1.5         0.2        0
4          5.0         3.6          1.4         0.2        0

ACCURACY OF THE MODEL:  0.9619047619047619
petal length (cm)    0.434564
petal width (cm)     0.420915
sepal length (cm)    0.095083
sepal width (cm)     0.049438
dtype: float64
  
===============================================================================================================

# K-nearest Neighbour Algorithm:

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.model_selection import train_test_split 

from sklearn.datasets import load_iris 

import numpy as np 

import matplotlib.pyplot as plt 

irisData = load_iris() 

X = irisData.data 

y = irisData.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)  

neighbors = np.arange(1, 9) 

train_accuracy = np.empty(len(neighbors)) 

test_accuracy = np.empty(len(neighbors)) 

for i, k in enumerate(neighbors): 

    knn = KNeighborsClassifier(n_neighbors=k) 

    knn.fit(X_train, y_train) 

    train_accuracy[i] = knn.score(X_train, y_train) 

    test_accuracy[i] = knn.score(X_test, y_test) 

plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 

plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 

plt.legend() 

plt.xlabel('n_neighbors') 

plt.ylabel('Accuracy') 

plt.show()

# Output:

===================================================================================================


