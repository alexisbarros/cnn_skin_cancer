import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# carregar dataset
skin_cancer_dataset = np.loadtxt('dataset/skin_cancer_dataset.txt')

# Selecionando as melhores features
X = SelectKBest(score_func=chi2, k=45000).fit_transform(skin_cancer_dataset[:, 1:], skin_cancer_dataset[:, 0])

# separar dados para treino e validacao
X_train, X_test, y_train, y_test = train_test_split(X, skin_cancer_dataset[:, 0], random_state=0)

# criar modelo
model = pickle.load(open('./trained_model_svm_selctKBest.sav', 'rb'))

# variaveis falso e verdadeiro, positivo e negativo
tp = 0
fn = 0
tn = 0
fp = 0

# predizer
predictedclasses = model.predict(X_test)

index = 0
for flag in y_test:
    if flag == 0:
        if predictedclasses[index] == 0:
            tn += 1
        else:
            fp += 1

    else:
        if predictedclasses[index] == 1:
            tp += 1
        else:
            fn += 1

    index += 1

# imprimir resultados
print('tp', tp)
print('tn', tn)
print('fp', fp)
print('fn', fn)
print('----')
print('sensitivity: ', "%.2f" % (tp/(tp+fn)))
print('specificity: ', "%.2f" % (tn/(tn+fp)))
