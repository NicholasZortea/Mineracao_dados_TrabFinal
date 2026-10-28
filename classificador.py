"""
Etapa 4 - Classificacao com Random Forest.

Entrada: work_padronizado.csv (gerado por limpeza_dados.py).

O modelo e avaliado com validacao cruzada estratificada repetida
(4 folds x 10 repeticoes). As previsoes de todas as repeticoes sao acumuladas
para montar uma unica matriz de confusao e um unico relatorio por classe.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = BASE_DIR / "work_padronizado.csv"

COLUNA_CLASSE = "Class"

NUMERO_DE_FOLDS = 4
NUMERO_DE_REPETICOES = 10
SEMENTE_VALIDACAO = 42

NUMERO_DE_ARVORES = 300
SEMENTE_MODELO = 77


def carregar_dados():
    """Le o dataset padronizado e separa atributos (X) e classe (y)."""
    df = pd.read_csv(ARQUIVO_ENTRADA)
    X = df.drop(COLUNA_CLASSE, axis=1)
    y = df[COLUNA_CLASSE]
    return X, y


def criar_modelo():
    """Random Forest com pesos balanceados, pois as classes tem tamanhos diferentes."""
    return RandomForestClassifier(
        n_estimators=NUMERO_DE_ARVORES,
        class_weight="balanced",
        random_state=SEMENTE_MODELO,
    )


def executar_validacao_cruzada(X, y):
    """
    Treina e testa o modelo em cada divisao da validacao cruzada.

    Devolve as classes reais e previstas acumuladas de todas as divisoes e a
    acuracia obtida em cada uma delas.
    """
    validacao = RepeatedStratifiedKFold(
        n_splits=NUMERO_DE_FOLDS,
        n_repeats=NUMERO_DE_REPETICOES,
        random_state=SEMENTE_VALIDACAO,
    )
    modelo = criar_modelo()

    classes_reais = []
    classes_previstas = []
    acuracias = []

    for indices_treino, indices_teste in validacao.split(X, y):
        X_treino, y_treino = X.iloc[indices_treino], y.iloc[indices_treino]
        X_teste, y_teste = X.iloc[indices_teste], y.iloc[indices_teste]

        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)

        classes_reais.extend(y_teste)
        classes_previstas.extend(previsoes)
        acuracias.append(accuracy_score(y_teste, previsoes))

    return classes_reais, classes_previstas, acuracias


def imprimir_matriz_de_confusao(classes_reais, classes_previstas, classes):
    print(f"Matriz de Confusao (acumulada das {NUMERO_DE_REPETICOES} repeticoes da validacao cruzada):")
    print("Linhas = classe real | Colunas = classe prevista")

    matriz = confusion_matrix(classes_reais, classes_previstas, labels=classes)
    tabela = pd.DataFrame(matriz, index=classes, columns=classes)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tabela)


def imprimir_relatorio(classes_reais, classes_previstas, acuracias):
    print("\nRelatorio de Classificacao por Classe (validacao cruzada):")
    # zero_division=0 evita erros caso alguma classe nunca seja prevista
    print(classification_report(classes_reais, classes_previstas, zero_division=0))

    acuracia_media = sum(acuracias) / len(acuracias)
    print(f"Acuracia media por divisao: {acuracia_media:.3f}")


def classificar():
    """Executa a etapa completa e imprime os resultados."""
    X, y = carregar_dados()
    classes = sorted(y.unique())

    classes_reais, classes_previstas, acuracias = executar_validacao_cruzada(X, y)

    imprimir_matriz_de_confusao(classes_reais, classes_previstas, classes)
    imprimir_relatorio(classes_reais, classes_previstas, acuracias)


if __name__ == "__main__":
    classificar()
