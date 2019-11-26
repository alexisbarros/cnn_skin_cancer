from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# carregar dataset
skin_cancer_dataset = np.loadtxt('skin_cancer_dataset.txt')

# Selecionando as melhores features
print(skin_cancer_dataset[:, 1:].shape)
X = SelectKBest(score_func=chi2, k=48000).fit_transform(skin_cancer_dataset[:, 1:], skin_cancer_dataset[:, 0])
print(X.shape)

# separar dados para treino e validacao
X_train, X_test, y_train, y_test = train_test_split(X, skin_cancer_dataset[:, 0], random_state=0)

# treinar modelo
model = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=500, verbose=1)
model.fit(X_train, y_train)

# predizer dados de validacao
predictedclasses = model.predict(X_test)

# imprimir resultados
print('Predicted Instances')
print(predictedclasses)

print('Test Real Values')
print(y_test)

classifier_accuracy = np.mean(predictedclasses==y_test)
print('Classifier Accuracy')
print(classifier_accuracy)

# salvar modelo
pickle.dump(model, open('trained_model_nn.sav', 'wb'))
