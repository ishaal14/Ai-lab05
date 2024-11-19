import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Dataset
data = {
"Color": ["Red", "Red", "Red", "Yellow", "Yellow", "Yellow", "Yellow", "Yellow", "Red", "Red"],
"Type": ["Sports", "Sports", "Sports", "Sports", "Sports", "SUV", "SUV", "SUV", "SUV", "Sports"],
"Origin": ["Domestic", "Domestic", "Domestic", "Domestic", "Domestic", "Imported", "Imported", "Domestic", "Imported", "Imported"],
"Stolen?": ["Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes"]
}

#Convert data into DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
label_encoder = LabelEncoder()
df["Color"]= label_encoder.fit_transform(df["Color"])
df ["Type"] = label_encoder.fit_transform(df["Type"])
df ["Origin"] = label_encoder.fit_transform(df["Origin"])
df["Stolen?"] = label_encoder.fit_transform(df ["Stolen?"]) #1 for 'Yes', 0 for 'No'

#Split data into features and target
x = df[["Color", "Type", "Origin"]]
y = df ["Stolen?"]

#Train/Test split
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

# Decision Tree rules
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\n Descison Tree Rules:\n", tree_rules)

