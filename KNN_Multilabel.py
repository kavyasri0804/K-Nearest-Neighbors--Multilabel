import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("iris.csv")
df=df.drop(['Id'],axis=1)
print(df)
df['Species']=df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}).astype(int)
X=df.iloc[ : ,:-1]
y=df.iloc[ : ,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
import warnings
warnings.filterwarnings("ignore")
reg=KNeighborsClassifier()
reg.fit(X_train,y_train)

#Train Performance
y_train_prediction=reg.predict(X_train)
print(f"{confusion_matrix(y_train,y_train_prediction)}")
print(f"Accuracy:{accuracy_score(y_train,y_train_prediction)}")
print(f"{classification_report(y_train,y_train_prediction)}")

#Test Performance
y_test_prediction=reg.predict(X_test)
print(f"{confusion_matrix(y_test,y_test_prediction)}")
print(f"Accuracy:{accuracy_score(y_test,y_test_prediction)}")
print(f"{classification_report(y_test,y_test_prediction)}")

#Own Datapoint Prediction
big_leaf_length=5.6
big_leaf_width=4.2
small_leaf_length=3.5
small_leaf_width=4.7
result=reg.predict([[big_leaf_length,big_leaf_width,small_leaf_length,small_leaf_width]])
if result==0:
    print(f"Given tree leaf was Setosa")
elif result==1:
    print(f"Given tree leaf was Versicolor")
else:
    print(f"Given tree leaf was Virginica")

with open("iris_model.pkl","wb") as f:
    pickle.dump(reg,f)
with open("iris_model.pkl","rb") as p:
    m=pickle.load(p)

result=m.predict([[big_leaf_length,big_leaf_width,small_leaf_length,small_leaf_width]])
if result==0:
    print(f"Given tree leaf was Setosa")
elif result==1:
    print(f"Given tree leaf was Versicolor")
else:
    print(f"Given tree leaf was Virginica")