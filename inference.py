import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load('trained_model.pkl')

# Load the test data
test_df = pd.read_csv('data/test.csv')
test_df['gender']=pd.factorize(test_df['gender'])[0]
cleaned_test_df = test_df.dropna()

# Make predictions
predictions = model.predict(cleaned_test_df)

# Load the true target labels
true_labels_df = pd.read_csv('data/target_labels_for_accuracy.csv')
cleaned_true_labels_df = true_labels_df.dropna()

# Calculate accuracy
accuracy = accuracy_score(cleaned_true_labels_df, predictions)

print(f"Inference result: {predictions.tolist()}")
print(f"Accuracy: {accuracy}")
