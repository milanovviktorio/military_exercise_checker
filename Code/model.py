import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib as jb

df = pd.read_csv("./Files/exercise_angles.csv") 

print(df.head())
print(df.columns)

X = df.drop(["Label", "Side"], axis=1)   # angle
y = df["Label"]                # exercise name

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X.head())
print(X.columns)

# ------------ MODEL EVALUATIONS ------------
# models = {
#     "RandomForest": RandomForestClassifier(),
#     "SVM": SVC(),
#     "KNN": KNeighborsClassifier(),
#     "GradientBoosting": GradientBoostingClassifier()
# }

# for name, m in models.items():
#     m.fit(X_train, y_train)
#     acc = m.score(X_test, y_test)
#     print(name, acc)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

jb.dump(model, "./Files/model.pkl")