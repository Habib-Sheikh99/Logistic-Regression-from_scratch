from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target)

scaler=StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X))

xtrain, xtest, ytrain, ytest = train_test_split(
    X_scaled, y, 
    test_size=0.25,
    random_state=42
    )

system=Logistic_Regression()
system.fit(xtrain, ytrain)

ypred = system.predict(xtest)

print(f"Accuracy : {accuracy_score(ytest, ypred)}")
# Accuracy : 0.951048951048951 
#          ~ 95%
