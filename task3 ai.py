#Task 3
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


# Data: ['Gender', 'Height', 'Weight', 'Foot_Size']
data = np.array([
    [1, 6.00, 180, 12],  # male
    [1, 5.92, 190, 11],  # male
    [1, 5.58, 170, 12],  # male
    [1, 5.92, 165, 10],  # male
    [0, 5.00, 100, 6],   # female
    [0, 5.50, 150, 8],   # female
    [0, 5.42, 130, 7],   # female
    [0, 5.75, 150, 9]    # female
])

X = data[:, 1:]  
y = data[:, 0]   
clf = DecisionTreeClassifier()
clf.fit(X, y)

new_entry = np.array([[5.5, 160, 10]])

predicted_gender = clf.predict(new_entry)[0]
predicted_gender_label = 'male' if predicted_gender == 1 else 'female'

print(f"Predicted gender for the new entry: {predicted_gender_label}")

new_entry_with_gender = np.hstack([np.array([[predicted_gender]]), new_entry])  
updated_data = np.vstack([data, new_entry_with_gender])  

X_updated = updated_data[:, 1:]  
y_updated = updated_data[:, 0]   
clf.fit(X_updated, y_updated)

print("\nUpdated dataset after adding the new entry:")
print(updated_data)
