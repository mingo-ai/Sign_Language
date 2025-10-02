from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import joblib

df = pd.read_csv("asl_keypoints.csv")
print(df.head())

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=11)

clf = RandomForestClassifier(bootstrap= False, n_estimators=250, max_depth= 25, max_features= 'sqrt', max_leaf_nodes=500, min_samples_leaf=2, min_samples_split=5,random_state=21) # max_samples= 1500,

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(clf, "gesture_model.pkl")
print("Model saved to gesture_model.pkl")  



# run python train_classifier.py

