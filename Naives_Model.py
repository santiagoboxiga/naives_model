import numpy as np
import csv

# Path to your data file
file_path = "/Users/santi/OneDrive/Santiago/Universidad/Curtin/Planning and Handling Uncertainty in ML/Assignment/Database_Churn.txt"

# Load data, skipping the first column and header
data = []
with open(file_path, newline='', encoding='utf-8') as csvfile:
    data_reader = csv.reader(csvfile, delimiter=',')
    next(data_reader)  # Skip the header
    for row in data_reader:
        converted_row = []
        for item in row[1:10]:  # Only take columns 2 to 10 (excluding 'Customer ID')
            if item.isdigit():
                converted_row.append(int(item))  # Convert to integer
            else:
                converted_row.append(-1)  # Replace non-integers with -1
        data.append(converted_row)

data = np.array(data)

# Shuffle the data and split into training (75%) and testing (25%)
np.random.shuffle(data)
split_index = int(0.75 * len(data))  # 75% for training, 25% for testing
train_data = data[:split_index]  # First 75% for training
test_data = data[split_index:]   # Remaining 25% for testing

# Separate features and labels
X_train = train_data[:, :-1]  # Features of training data
y_train = train_data[:, -1]   # Labels (Churn) of training data
X_test = test_data[:, :-1]    # Features of test data
y_test = test_data[:, -1]     # Labels (Churn) of test data

# Train Naive Bayes
classes, class_counts = np.unique(y_train, return_counts=True)
class_prob = {}
for idx, class_val in enumerate(classes):
    class_prob[class_val] = class_counts[idx] / class_counts.sum()

feature_prob = {}
for feature in range(X_train.shape[1]):
    feature_prob[feature] = {}
    for class_val in classes:
        feature_prob[feature][class_val] = {}
        feature_subset = X_train[y_train == class_val]
        values, counts = np.unique(feature_subset[:, feature], return_counts=True)
        total = counts.sum()
        for idx, value in enumerate(values):
            feature_prob[feature][class_val][value] = counts[idx] / total

# Make predictions
predictions = []
for instance in X_test:
    instance_probs = {}
    for class_val in class_prob:
        prod_prob = class_prob[class_val]
        for feature in range(len(instance)):
            feature_value = instance[feature]
            if feature_value in feature_prob[feature][class_val]:
                prod_prob *= feature_prob[feature][class_val][feature_value]
            else:
                prod_prob *= 1e-10  # Small value to avoid zero probability
        instance_probs[class_val] = prod_prob
    
    # Find the class with the highest probability
    predicted_class = None
    max_prob = -1
    for class_val, prob in instance_probs.items():
        if prob > max_prob:
            max_prob = prob
            predicted_class = class_val
    predictions.append(predicted_class)

# Calculate accuracy
correct = 0
total = len(y_test)
for idx, prediction in enumerate(predictions):
    if prediction == y_test[idx]:
        correct += 1
accuracy = correct / total

print("Number of correct predictions:", correct)
print("Number of total predictions:" , total)
print("Accuracy of the model:", accuracy)
