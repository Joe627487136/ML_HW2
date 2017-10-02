import numpy as np
import pandas as pd
# machine learning libraries
from sklearn.linear_model import LogisticRegression


X_data=pd.read_csv("train.csv")
X_test=pd.read_csv("test.csv")
X_valid=X_data.sample(frac=0.2,random_state=200)
X_train=X_data.drop(X_valid.index)
Y_data=X_data["Survived"]
Y_valid=X_valid["Survived"]
Y_train=X_train["Survived"]
ID_test=X_test["PassengerId"]

from IPython.display import display
'''
display(X_data.head())
display(X_data.describe())
display(X_test.head())
display(X_test.describe())
'''



#df=X_train


def preprocess(df):
    df.drop(["Survived"], axis=1, inplace=True, errors="ignore")
    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)

    df = df.join(pd.get_dummies(df["Embarked"]))
    df.drop(["Embarked"], axis=1, inplace=True)
    df = df.join(pd.get_dummies(df["Sex"]))
    df.drop(["Sex"], axis=1, inplace=True)
    df = df.join(pd.get_dummies(df["Pclass"]))
    df.drop(["Pclass"], axis=1, inplace=True)

    df.loc[(df['SibSp']>0)|(df['Parch']>0), "Family"] = 1
    df.loc[(df['SibSp']==0)&(df['Parch']==0), "Family"] = 0
    df.loc[df['Age'] <= 16, "Child"] = 1
    df.loc[df['Age'] > 16, "Child"] = 0

    return df


X_train=preprocess(X_train)
X_valid=preprocess(X_valid)
X_data=preprocess(X_data)
X_test=preprocess(X_test)
display(X_train.head())

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_valid, Y_valid) * 100, 2)

print(acc_log)


nlogreg = LogisticRegression()
nlogreg.fit(X_data,Y_data)
print(nlogreg.coef_)


Y_test = nlogreg.predict(X_test)
ans=pd.DataFrame({"PassengerId":ID_test,"Survived":Y_test})
ans.to_csv("submit.csv",index=False)


