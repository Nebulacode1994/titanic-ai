import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df.drop(['Name','Ticket','Cabin','PassengerId'], axis = 1 )

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Sex'] = df['Sex'].map({'male': 0 , 'female': 1})

df['Embarked'] = df['Embarked'].fillna('S')

df = pd.get_dummies(df, columns = ['Embarked'])

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

df = df.drop(['SibSp','Parch'], axis = 1)

x = df.drop('Survived' , axis = 1)

y = df['Survived']

x_train,y_train,x_test,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

model = RandomForestClassifier(n_estimator = 100)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = model.score(y_test,y_pred)

print(f"AI model grade: {accuracy * 100:.2f}%")


