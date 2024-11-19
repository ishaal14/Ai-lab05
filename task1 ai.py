import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from the uploaded Excel file
file_path = 'lab5 datasett.xlsx'  # Replace with the correct path to your dataset
df = pd.read_excel(file_path)

# Prepare features and labels
features = df[['Height', 'Weight', 'Foot_Size']]
labels = df['Gender']

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier()

# Train the model
classifier.fit(features, labels)

# Define a new entry to classify
new_entry = [[6.00, 180, 12]]

# Predict the class of the new entry
prediction = classifier.predict(new_entry)
print("Predicted Gender:", prediction[0])
