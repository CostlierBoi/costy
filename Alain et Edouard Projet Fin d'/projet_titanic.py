import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

df = pd.read_csv('titanic.csv')
df = df.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
df = df.dropna()
data = pd.get_dummies(df)

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(scaled_X_train,y_train) 

y_pred_test = knn_model.predict(scaled_X_test)
test_error = accuracy_score(y_test,y_pred_test)
RMSE = (mean_squared_error(y_test,y_pred_test))**0.5
print(test_error,RMSE)

def predict_survival(data=list):
    return knn_model.predict([data])