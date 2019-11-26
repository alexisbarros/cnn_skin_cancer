import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import tkinter

# carregar dataset
skin_cancer_dataset = np.loadtxt('dataset/skin_cancer_dataset.txt')

# Selecionando as melhores features
X = SelectKBest(score_func=chi2, k=48000).fit_transform(skin_cancer_dataset[:, 1:], skin_cancer_dataset[:, 0])

# separar dados para treino e validacao
X_train, X_test, y_train, y_test = train_test_split(X, skin_cancer_dataset[:, 0], random_state=0)

# criar modelo
model = pickle.load(open('./trained_model_nn.sav', 'rb'))

# predizer
predictedclasses = model.predict(X_test)

# calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_test, predictedclasses)

# plotar curva ROC
plt.plot(fpr, tpr)
plt.axis([0, 1, 0, 1])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.show()

auc_score = roc_auc_score(y_test, predictedclasses)
print(auc_score)
