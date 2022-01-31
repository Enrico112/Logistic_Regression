import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('insurance.csv')

plt.scatter(df.age, df.bought_insurance, marker='o', color='blue')

# split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=.8)

# logistic regression obj
model = LogisticRegression()
# train model
model.fit(X_train,y_train)
# predict
model.predict(X_test)
# predict for 56yo
model.predict([[56]])
# accuracy
model.score(X_test, y_test)

# predict probability of x being in a certain y class
model.predict_proba(X_test)
# predict prob for 56yo
model.predict_proba([[56]])


