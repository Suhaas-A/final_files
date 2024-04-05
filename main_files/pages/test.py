import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for saving the model

# Assuming you have your features and labels in 'sign_language_data.csv'
df = pd.read_csv('data.csv')

# Split data into features (X) and labels (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict on the test set
predictions = clf.predict(X_test)

# Print the accuracy
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Save the model to a file
joblib.dump(clf, 'sign_language_model.pkl')
print("Model saved as 'sign_language_model.pkl'.")