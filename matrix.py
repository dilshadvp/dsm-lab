#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
a=np.array([[2,4],[3,6]])
b=np.array([[1,2],[4,8]])
c=a+b
d=a-b;
e=a*b;
f=np.dot(a,b)
g=np.transpose(a)
print("addition\n",c);
print("substraction\n",d);
print("multiplication\n",e);
print("transpose mul\n",f);
print("transpose org\n",g);


# In[13]:


import numpy as np
x=np.array([[1,2,3],[4,5,6],[7,8,9]])
U,S,VT=np.linalg.svd(x)
n_components=2
x_reconstructed=np.dot(U[:,:n_components],np.dot(np.diag(S[:n_components]),VT[:n_components,:]))
print("original matrix:")
print(x)
print("\nreconstructed matrix(with reduced dimensions):")
print(x_reconstructed)


# In[21]:


A=np.array([[4,11],[14,5]])
U,s,Vt=np.linalg.svd(A)
Sigma=np.diag(s)
print("---matrix U (singular vectors)---")
print(np.round(U,4))
print("\n--matrix sigma(singular value diagonal)---")
print(np.round(sigma,4))
print("\n---matrix Vt (Right singular vector transposed)---")
print(np.round(Vt,4))
A_reconstructed = U @ Sigma @ Vt
print("\n---Verification(U * Sigma * Vt)---")
print(np.round(A_reconstructed,4))


# In[23]:


import matplotlib.pyplot as plt
X=[3,5,6,8,9]
Y=[1,2,3,4,5]
plt.plot(X,Y)
plt.title("DILSHAD")
plt.xlabel("population")
plt.ylabel("growth")


# In[25]:


import matplotlib.pyplot as plt
subject=["maths","physics","chemistry","biology"]
mark=[60,70,80,90]
plt.bar(subject,mark)
plt.title("students mark")
plt.xlabel("subjects")
plt.ylabel("marks")


# In[32]:


import matplotlib.pyplot as plt
plt.hist(X)
plt.title("DILSHAD")
plt.xlabel("population")
plt.ylabel("growth")
X=[10,10,10,20,30,30,40,40]


# In[33]:


import matplotlib.pyplot as plt
plt.hist(X)
plt.title("DILSHAD")
plt.xlabel("population")
plt.ylabel("growth")
X=[10,10,10,20,30,30,40,40]
plt.legend("pele")


# In[42]:


import matplotlib.pyplot as plt
plt.title("DILSHAD")
Y=[20,30,40,50]
plt.pie(Y)
plt.show
plt.legend("BYGR")


# In[50]:


import matplotlib.pyplot as plt
X=[1,2,6,18]
Y=[3,10,12,20]
plt.plot(X,Y,'r:o')
plt.show
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend("line")


# In[57]:


import matplotlib.pyplot as plt
X=[1,2,3,4]
Y=[5,6,7,8]
Z=[2,3,4,5]
plt.plot(X,Y,Z)
plt.show()


# In[59]:


import matplotlib.pyplot as plt
plt.hist(X)
plt.title("SCORES")
plt.xlabel("men")
plt.ylabel("women")
x=np.array([0,1,2,3])
y=np.array([3,8,1,10])
plt.subplot(1,2,1)
plt.plot(x,y)


# In[24]:


import matplotlib.pyplot as plt
plt.title("SCORES")
plt.xlabel("group")
plt.ylabel("scores")

men=(26,30,35,35,26)
women=(25,32,30,35,29)

gender=['G1','G2','G3','G4','G5']

plt.bar(gender,men,label="men")
plt.bar(gender,women,label="women")
plt.legend()
plt.show()


# In[27]:


import matplotlib.pyplot as plt
group=[1,2,3,4,5]
men=(22,30,35,35,36)
women=(25,32,30,35,29)
plt.bar(group,men,label="men")
plt.bar(group,women,label="women")
plt.xlabel("group")
plt.ylabel("scores")
plt.legend()
plt.show()


# In[29]:


import matplotlib.pyplot as plt
language=['java','python','php','js','C#','C++']
popularity=[22.2,17.6,8.8,7.7,6.7]
plt.pie(popularity)
plt.title("pie chart")


# In[32]:


import matplotlib.pyplot as plt
import numpy as np
language=['java','python','php','js','C#','C++']
popularity=[22.2,17.6,8.8,7.7,6.7,5.7]
colors=np.array(["green","red","orange","blue","purple","yellow"])
plt.scatter(language, popularity,c=colors)
plt.title("scatter")


# In[33]:


import matplotlib.pyplot as plt
import numpy as np
language=['java','python','php','js','C#','C++']
popularity=[22.2,17.6,8.8,7.7,6.7,5.7]
plt.barh(language, popularity)
plt.title("scatter")


# In[57]:


import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4,5])
y=np.array([5,6,7,8,9])
plt.subplot(2,2,1)
plt.plot(x,y,'o')



y1=np.array([2,4,6,8,9])
y2=np.array([5,4,3,2,1])
plt.subplot(2,2,2)
plt.plot(y1,linestyle='dashed',linewidth='2.5')
plt.plot(y2,linewidth='2.5')


y3=np.array([1,2,6,8,9])
y4=np.array([5,3,4,2,7])
plt.subplot(2,2,3)
plt.bar(y3,y4)
plt.plot(y4,linewidth='2.5')
plt.show()


# In[4]:


from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
# print(X,y)
print(iris.feature_names)
print(iris.target_names)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)
print(f'Accuracy:{accuracy:.2f}')
print(y_predict)
new=[[2,3,4,5]]
new_predict=knn.predict(new)
print(iris.target_names[new_predict])


# In[9]:


import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
iris=load_breast_cancer()
X=iris.data
y=iris.target
# print(X,y):3
print(iris.feature_names)
# print(iris.target_names)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)
print(f'Accuracy:{accuracy:.2f}')
print(y_predict)
# new=[[2,3,4,5]]
# new_predict=knn.predict(new)
# print(iris.target_names[new_predict])
new = [[17.99, 10.38, 122.80, 1001.0, 0.11840,
    0.27760, 0.30010, 0.14710, 0.2419, 0.07871,
    1.0950, 0.9053, 8.589, 153.40, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.60, 2019.0, 0.16220,
    0.66560, 0.71190, 0.26540, 0.4601, 0.11890]]
new_predict = knn.predict(new)
print(new_predict)


# In[2]:


from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_csv("insurance.csv")
data


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
import pandas as pd 
data=pd.read_csv("food.csv")
X = data.iloc[:, :3] 
print(X)
print("\n")
y= data.iloc[:, 3]
print(y) 
le = LabelEncoder() 
categorical_columns = ['Ingredient']   
for col in categorical_columns: 
    X[col] = le.fit_transform(X[col]) 
y = le.fit_transform(y)
print(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42) 
k=3 
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train,y_train) 
y_pred = knn.predict(X_test) 
sample=[[1,10,9]] 
k=knn.predict(sample) 
print("\n")
print(k) 
accuracy = accuracy_score(y_test,y_pred) 
print("/n")
print(accuracy) 


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
data=pd.read_csv("insurance.csv")
X = data.iloc[:, :6] 
print(X) 
y= data.iloc[:, 1] 
le = LabelEncoder() 
categorical_columns = ['sex', 'smoker', 'region']   
for col in categorical_columns: 
    X[col] = le.fit_transform(X[col]) 
y = le.fit_transform(y) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42) 
k=3 
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train,y_train) 
y_pred = knn.predict(X_test) 
sample=[[1,10,9,11,4,2]] 
k=knn.predict(sample) 
print(k) 
accuracy = accuracy_score(y_test,y_pred) 
print(accuracy)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
data=pd.read_csv("cricket.csv")
X = data.iloc[:, :5] 
print(X) 
y= data.iloc[:, 4] 
print(y) 
le = LabelEncoder() 
categorical_columns = ['Outlook','Temp','Humidity','Windy','Play Cricket']   
for col in categorical_columns: 
    X[col] = le.fit_transform(X[col]) 
y = le.fit_transform(y) 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42) 
k=3 
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test) 
sample=[[1,10,9,11,4]] 
k=knn.predict(sample) 
print(k) 
accuracy = accuracy_score(y_test,y_pred) 
print(accuracy)


# In[20]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
nb_classifier=GaussianNB()
nb_classifier.fit(X_train,y_train)
y_pred=nb_classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
new=[[1,2,3,4]]
new_predict=nb_classifier.predict(new)
print(iris.target_names[new_predict])
print(f"Accuracy:{accuracy*100:2f}%")


# In[ ]:




