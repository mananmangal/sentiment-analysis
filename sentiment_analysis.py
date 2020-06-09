import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
with open("train.dat",'r') as f1:
    text_train = f1.readlines()
y = np.loadtxt('train.labels')
#y[y==-1] = 0
with open("test.dat",'r') as f2:
    text_test = f2.readlines()
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
X = vectorizer.fit_transform(text_train)
test = vectorizer.transform(text_test)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def logistic_regression(X, y, num_steps, alpha):
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))
    theta = np.zeros(X.shape[1])
    for i in range(num_steps):
        z = np.dot(X, theta)
        hyp = sigmoid(z)
        y[y==-1] = 0
        gradient = np.dot(X.T, (hyp - y)) / y.size 
        #if (i%5000 ==0) and alpha > 0.00001:
         #   alpha = alpha/10
        theta = theta - (alpha * gradient)
    return theta
#X = np.load('train.mat')
#test = np.load('test.mat')
y = np.loadtxt('train.labels')
theta = logistic_regression(X.toarray(), y, num_steps = 1000, alpha = 0.1)
finalTheta = np.dot(np.hstack((np.ones((test.shape[0], 1)), test.toarray())), theta)
preds = np.round(sigmoid(finalTheta))
preds[preds==0] = -1
preds[preds==1] = 1
np.savetxt('test.dat',preds)