print("ML solution proposed by : VAIBHAV CHAURASIA")
print("\nEmail ID : vaibhav24012000@gmail.com")






import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import  DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings(action="ignore")

filename="HeartAttack_data.csv"
data=pd.read_csv(filename,index_col=False)
df=pd.DataFrame(data)
print("First five data = \n\n",df.head())

print("Shape of the data",df.shape)
print("Size= ",df.size)

print("Data Description =",df.describe())


df.replace('?',np.nan,inplace=True)
print(df)
df=df.groupby(df.columns,axis=1).transform(lambda x:x.fillna(x.median()))
print(df.isnull().sum())


print(df.head(10))

print("\n\n\ndata.groupby('num').size()\n")
print(data.groupby('num').size())

plt.hist(df['num'])
plt.title("num PLOT")

print(df)
df.plot(kind="density",subplots=True,layout=(3,4),sharex=False)

plt.show()
array=df.values
X=array[:,0:12]
Y=array[:,13]
Y=Y.astype('float64')

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.33,random_state=25)





pipeline=[]


pipeline.append(('ScaledCart',Pipeline([('Scaler',StandardScaler()),('Cart',DecisionTreeClassifier())])))
pipeline.append(('ScaledSVM',Pipeline([('Scaler',StandardScaler()),('SVM',SVC())])))
pipeline.append(('ScaledNB',Pipeline([('Scaler',StandardScaler()),('Gaussian NB',GaussianNB())])))
pipeline.append(('ScaledKNN',Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsClassifier())])))






num_fold=10
names=[]
result=[]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    kfold=KFold(n_splits=num_fold,random_state=150)
    for name,model in pipeline :

        startTime=time.time()
        cv_result=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
        endTime=time.time()
        result.append(cv_result)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_result.mean(), cv_result.std(), endTime-startTime))



fig=plt.figure()
fig.title=("Performance Chart")
ax=fig.add_subplot(111)
plt.boxplot(result)
ax.set_xticklabels(names)
plt.show()



scalar=StandardScaler().fit(X_train)

scaledX_train=scalar.transform(X_train)
model=SVC()
start=time.time()
model.fit(scaledX_train,Y_train)
end=time.time()



scaledX_test=scalar.transform(X_test)
prediction=model.predict(scaledX_test)


print("\n\nAccuracy = %f" % (accuracy_score(Y_test,prediction)))

print("\n\n")
print("CONFUSION MATRIX = ")

print(confusion_matrix(Y_test,prediction))






print("\n\n\nML solution proposed by : VAIBHAV CHAURASIA")
print("\nEmail ID : vaibhav24012000@gmail.com")

