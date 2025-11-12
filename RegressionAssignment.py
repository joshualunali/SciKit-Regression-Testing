import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pandas.read_csv("RegressionData.csv", header = None, names=['X', 'y']) # 5 points

X = data['X'].values.reshape(-1,1) 
y = data['y'] 
plt.plot("X", "y", "ro", data=data ) 
plt.show()

reg = linear_model.LinearRegression() 
reg.fit(X, y) 

fig = plt.figure()
y_pred = reg.predict(X) 
plt.scatter(X,y, c='b', marker='o') 
plt.plot(X, y_pred, 'r') 
fig.canvas.draw()
plt.show()


data = pandas.read_csv("LogisticRegressionData.csv", header = None, names=['Score1', 'Score2', 'y']) # 2 points


X = data.drop(['y'],axis=1) 
y = data['y'] 


m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(X['Score1'][i], X['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) # 2 points
fig.canvas.draw()
plt.show()


regS = linear_model.LogisticRegression() 
regS.fit(X, y) 


y_pred = regS.predict(X) 
m = ['o', 'x']
c = ['red', 'blue'] 
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(X['Score1'][i], X['Score2'][i], marker=m[y_pred[i]], c = c[y_pred[i]]) 
fig.canvas.draw()
plt.show()
