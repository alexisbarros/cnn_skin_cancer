from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load dataset
skin_cancer_dataset = np.loadtxt('dataset/skin_cancer_dataset.txt')

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    skin_cancer_dataset[:, 1:], skin_cancer_dataset[:, 0], random_state=0)

# Train model
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Predict test cases
predictedclasses = naive_bayes.predict(X_test)

# Print results
print('Predicted Instances')
print(predictedclasses)

print('Test Real Values')
print(y_test)

classifier_accuracy = np.mean(predictedclasses==y_test)
print('Classifier Accuracy')
print(classifier_accuracy)
