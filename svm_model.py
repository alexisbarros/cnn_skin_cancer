from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# carregar dataset
skin_cancer_dataset = np.loadtxt('dataset/skin_cancer_dataset.txt')

# Selecionando as melhores features
print(skin_cancer_dataset[:, 1:].shape)
X = SelectKBest(score_func=chi2, k=45000).fit_transform(skin_cancer_dataset[:, 1:], skin_cancer_dataset[:, 0])
print(X.shape)

# separar dados para treino e validacao
X_train, X_test, y_train, y_test = train_test_split(X, skin_cancer_dataset[:, 0], random_state=0)

# treinar modelo
lin_svc = svm.SVC(kernel='linear', C=1.0, verbose=1)
lin_svc.fit(X_train, y_train)

# predizer dados de validacao
predictedclasses = lin_svc.predict(X_test)

# imprimir resultados
print('Predicted Instances')
print(predictedclasses)

print('Test Real Values')
print(y_test)

classifier_accuracy = np.mean(predictedclasses==y_test)
print('Classifier Accuracy')
print(classifier_accuracy)

# salvar modelo
pickle.dump(lin_svc, open('trained_model_svm.sav', 'wb'))
