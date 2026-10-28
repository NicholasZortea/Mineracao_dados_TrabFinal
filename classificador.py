from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

df = pd.read_csv("work_padronizado.csv")
X = df.drop('Class', axis=1)
y = df['Class']

rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=42)

clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=77)


y_true_cv, y_pred_cv, acuracias = [], [], []
for train_idx, test_idx in rskf.split(X, y):
    clf.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = clf.predict(X.iloc[test_idx])
    y_true_cv.extend(y.iloc[test_idx])
    y_pred_cv.extend(y_pred)
    acuracias.append(accuracy_score(y.iloc[test_idx], y_pred))



print("Matriz de Confusão (Acumulada das 10 repetições da Validação Cruzada):")
print("Linhas = classe real | Colunas = classe prevista")
classes = sorted(y.unique())
cm = pd.DataFrame(confusion_matrix(y_true_cv, y_pred_cv, labels=classes), index=classes, columns=classes)
with pd.option_context('display.width', 200, 'display.max_columns', None):
    print(cm)

print("\nRelatório de Classificação por Classe (Validação Cruzada):")
# zero_division=0 evita erros caso alguma classe tenha comportamento atípico
print(classification_report(y_true_cv, y_pred_cv, zero_division=0))
