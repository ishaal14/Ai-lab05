# Task 2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

file_path = 'lab5 datasett.xlsx'  
df = pd.read_excel(file_path)


features = df[['Height', 'Weight', 'Foot_Size']]
labels = df['Gender']


classifier = DecisionTreeClassifier()

classifier.fit(features, labels)

new_entry = [[6.00, 180, 12]]


prediction = classifier.predict(new_entry)
predicted_gender = prediction[0]


print("Predicted Gender for new entry:", predicted_gender)


new_data = pd.DataFrame({
    'Gender': [predicted_gender],
    'Height': [new_entry[0][0]],
    'Weight': [new_entry[0][1]],
    'Foot_Size': [new_entry[0][2]]
})

print("\nNew entry added to the DataFrame:")
print(new_data)
