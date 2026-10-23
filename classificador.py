from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

df = pd.read_csv("work_padronizado.csv")
X = df.drop('Class', axis=1)
y = df['Class']

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

y_pred_cv = cross_val_predict(clf, X, y, cv=skf)


print(f"Acurácia Média Global: {accuracy_score(y, y_pred_cv):.4f}\n")

print("Matriz de Confusão (Acumulada da Validação Cruzada):")
print("Linhas = classe real | Colunas = classe prevista")
classes = sorted(y.unique())
cm = pd.DataFrame(confusion_matrix(y, y_pred_cv, labels=classes), index=classes, columns=classes)
with pd.option_context('display.width', 200, 'display.max_columns', None):
    print(cm)

print("\nRelatório de Classificação por Classe (Validação Cruzada):")
# zero_division=0 evita erros caso alguma classe tenha comportamento atípico
print(classification_report(y, y_pred_cv, zero_division=0))