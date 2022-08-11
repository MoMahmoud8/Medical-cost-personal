import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#from google.colab import files
#uploaded = files.upload()



data = pd.read_csv('insurance.csv')


#Preprocessing 
data["smoker"]= data["smoker"].apply(lambda x:0 if x=='no' else 1)
data["sex"]= data["sex"].apply(lambda x:0 if x=='male' else 1)
data_region = data["region"]
data_region=data_region.drop_duplicates()
data_region=data_region.tolist()
data_region

data["region"]=data.region.map(dict(southwest=0, southeast=1,northwest=2,northeast=3))
#Converted text fileds to numerical data

#print(data.head())

standarized_df = data.copy()
standarized_df.iloc[:,0] = preprocessing.normalize([standarized_df.iloc[:,0]])[0]
standarized_df.iloc[:,1] = preprocessing.normalize([standarized_df.iloc[:,1]])[0]
standarized_df.iloc[:,2] = preprocessing.normalize([standarized_df.iloc[:,2]])[0]
standarized_df.iloc[:,3] = preprocessing.normalize([standarized_df.iloc[:,3]])[0]
standarized_df.iloc[:,4] = preprocessing.normalize([standarized_df.iloc[:,4]])[0]
standarized_df.iloc[:,5] = preprocessing.normalize([standarized_df.iloc[:,5]])[0]
standarized_df.iloc[:,6] = preprocessing.normalize([standarized_df.iloc[:,6]])[0]

#print(standarized_df.head())

def train_validate_test_split(data_pd,testRatio=0.25, valRatio=0.375):
    np.random.seed(42)
    perm=np.random.permutation(data_pd.index)
    m=len(data_pd.index)
    testEnd = int(testRatio * m)
    validateEnd = int(valRatio * m) + testEnd
    testSet = data_pd.iloc[perm[:testEnd]]
    validateSet = data_pd.iloc[perm[testEnd:validateEnd]]
    trainSet = data_pd.iloc[perm[validateEnd:]]
    return trainSet, validateSet, testSet

# we can use 'data' to use the orginal data, or 'standarized_df' to use the normalized data
train,validate,test=train_validate_test_split(data)

X_train=train.loc[:, train.columns!='charges']
y_train = train[["charges"]]
y_train=(y_train.to_numpy()).reshape([503,])
X_validate = validate.loc[:, train.columns!='charges']
y_validate = validate[["charges"]]
y_validate=(y_validate.to_numpy()).reshape([501,])
X_test = test.loc[:, train.columns!='charges']
y_test = test["charges"]
y_test=(y_test.to_numpy()).reshape([334,])
y_train=y_train.astype('float64')
y_validate=y_validate.astype('float64')
y_test=y_test.astype('float64')

#print("y_train shape ",y_train.shape)
#print("y_validate shape ",y_validate.shape)
#print("y_test shape ",y_test.shape)


from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(random_state=1, max_iter=10000).fit(X_train, y_train)
#print("neural_network model model (mostafa)-->Validation-->",nn.score(X_validate,y_validate)*100)
y_pred=nn.predict(X_test)
print("\nneural_network model-->",nn.score(X_test, y_test)*100)

print("-----------------------------------------------------------")
print("-----------------------------------------------------------")


from sklearn.tree import DecisionTreeRegressor

DecisionTree =DecisionTreeRegressor()
DecisionTree = DecisionTree.fit(X_train, y_train)
y_predi=DecisionTree.predict(X_test)
#print("DecisionTree model  ",DecisionTree.score(X_validate,y_validate)*100)
print("DecisionTree model--> ",DecisionTree.score(X_test,y_test)*100)

print("-----------------------------------------------------------")
print("-----------------------------------------------------------")



from sklearn.linear_model import LinearRegression

modelLR = LinearRegression()
modelLR.fit(X_train, y_train)
y_pred = modelLR.predict(X_test)
modelLR.score(X_test, y_test)
#print("LinearRegression model-->",modelLR.score(X_validate,y_validate)*100)
print("LinearRegression model--> ",modelLR.score(X_test,y_test)*100)

print("-----------------------------------------------------------")
print("-----------------------------------------------------------")



from sklearn.ensemble import RandomForestRegressor

# n_estimators = 100
clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("RandomForestRegressor--> ",clf.score(X_test,y_test)*100)

# n_estimators = 100
clf = RandomForestRegressor(n_estimators=50)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("RandomForestRegressor--> ",clf.score(X_test,y_test)*100)

print("-----------------------------------------------------------")
print("-----------------------------------------------------------")




from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=10)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
#print("KNN model-->",neigh.score(X_validate,y_validate)*100)
print("KNN model-->",neigh.score(X_test,y_test)*100)
print("-----------------------------------------------------------")
print("-----------------------------------------------------------")


from sklearn import svm

regr = svm.SVR()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
#print("SVM model--> ",regr.score(X_validate,y_validate)*100)
print("SVM model--> ",regr.score(X_test,y_test)*100)

print("-----------------------------------------------------------")
print("-----------------------------------------------------------")


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
#print("KNN model (ahmed)-->Validation-->",knn.score(X_validate,y_validate)*100)
y_pred=knn.predict(X_test)
print("KNN model-->",knn.score(X_test, y_test)*100)
print("-----------------------------------------------------------")
print("-----------------------------------------------------------\n")


plt.style.use('ggplot')
y = data.charges
_ = pd.plotting.scatter_matrix(data, c = y, figsize = [8, 8],s=150, marker = 'D')
#plt.show()