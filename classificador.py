"""
Etapa 3 - Classificacao com Random Forest.

Entrada: work_padronizado.csv (gerado por limpeza_dados.py) e
         AgeOfMythology_noClass.csv (unidades a classificar).
Saida:   resultados no terminal, matriz_confusao.png, importancia_atributos.png
         e previsoes_noClass.csv.

O modelo e avaliado com validacao cruzada estratificada repetida
(4 folds x 10 repeticoes). As previsoes de todas as repeticoes sao acumuladas
para montar uma unica matriz de confusao e um unico relatorio por classe.

Depois, o modelo final e treinado com todas as unidades. Ele e usado para:
- medir a importancia de cada atributo (feature_importances_): quanto, em
  media nas arvores, as divisoes feitas com aquele atributo reduziram a
  impureza de Gini. Os valores somam 1. Essa medida tende a favorecer
  atributos com muitos valores distintos (como HP) em relacao aos discretos
  (como Age);
- classificar as unidades do noClass. Esse arquivo ja vem com valores
  numericos limpos, entao nao passa pelas etapas de limpeza.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = BASE_DIR / "work_padronizado.csv"
ARQUIVO_SEM_CLASSE = BASE_DIR / "AgeOfMythology_noClass.csv"
ARQUIVO_GRAFICO_MATRIZ = BASE_DIR / "matriz_confusao.png"
ARQUIVO_GRAFICO_IMPORTANCIA = BASE_DIR / "importancia_atributos.png"
ARQUIVO_PREVISOES = BASE_DIR / "previsoes_noClass.csv"

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

    return matriz


def desenhar_matriz_de_confusao(matriz, classes):
    """Salva a matriz de confusao acumulada da validacao cruzada como imagem."""
    figura, eixo = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matriz, display_labels=classes).plot(
        ax=eixo, cmap="Blues", colorbar=False, xticks_rotation=45)
    # Alinha os nomes inclinados pela ponta, para nao se sobreporem
    plt.setp(eixo.get_xticklabels(), ha="right", rotation_mode="anchor")

    eixo.set_title(f"Matriz de confusao (validacao cruzada, {NUMERO_DE_REPETICOES} repeticoes)")
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")

    figura.tight_layout()
    figura.savefig(ARQUIVO_GRAFICO_MATRIZ, dpi=150)
    plt.close(figura)

    print(f"Grafico salvo em: {ARQUIVO_GRAFICO_MATRIZ.name}")


def imprimir_relatorio(classes_reais, classes_previstas, acuracias):
    print("\nRelatorio de Classificacao por Classe (validacao cruzada):")
    # zero_division=0 evita erros caso alguma classe nunca seja prevista
    print(classification_report(classes_reais, classes_previstas, zero_division=0))

    acuracia_media = sum(acuracias) / len(acuracias)
    print(f"Acuracia media por divisao: {acuracia_media:.3f}")


def treinar_modelo_final(X, y):
    """Treina o modelo com todas as unidades (a avaliacao ja foi feita na validacao cruzada)."""
    modelo = criar_modelo()
    modelo.fit(X, y)
    return modelo


def calcular_importancia_dos_atributos(modelo, X):
    """Devolve a importancia de cada atributo, da maior para a menor."""
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    return importancias.sort_values(ascending=False)


def imprimir_importancia(importancias):
    print("\nImportancia dos atributos (Random Forest, reducao media de Gini):")
    for atributo, importancia in importancias.items():
        print(f"  {atributo:<12} {importancia:.3f}")


def desenhar_grafico_de_importancia(importancias):
    """Grafico de barras horizontais, com o atributo mais importante no topo."""
    # barh desenha de baixo para cima, entao a ordem e invertida
    importancias_em_ordem = importancias.iloc[::-1]

    figura, eixo = plt.subplots(figsize=(7, 4.5))
    barras = eixo.barh(importancias_em_ordem.index, importancias_em_ordem.values)
    eixo.bar_label(barras, fmt="%.3f", padding=3, fontsize=9)

    eixo.set_title("Importancia dos atributos (Random Forest)")
    eixo.set_xlabel("Reducao media da impureza de Gini")
    eixo.set_xlim(0, importancias.max() * 1.15)

    figura.tight_layout()
    figura.savefig(ARQUIVO_GRAFICO_IMPORTANCIA, dpi=150)
    plt.close(figura)

    print(f"Grafico salvo em: {ARQUIVO_GRAFICO_IMPORTANCIA.name}")


def classificar_unidades_sem_classe(modelo, colunas):
    """
    Preve a classe de cada unidade do noClass e a probabilidade dada a ela
    (fracao das arvores que votaram nessa classe).
    """
    df_sem_classe = pd.read_csv(ARQUIVO_SEM_CLASSE)
    X_novo = df_sem_classe[colunas]

    probabilidades = modelo.predict_proba(X_novo)

    df_previsoes = df_sem_classe.copy()
    df_previsoes["Class"] = modelo.predict(X_novo)
    df_previsoes["Probabilidade"] = probabilidades.max(axis=1).round(3)

    df_previsoes.to_csv(ARQUIVO_PREVISOES, index=False)

    print("\nPrevisoes para o noClass:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df_previsoes)
    print(f"Arquivo salvo em: {ARQUIVO_PREVISOES.name}")
    return df_previsoes


def classificar():
    """Executa a etapa completa e imprime os resultados."""
    X, y = carregar_dados()
    classes = sorted(y.unique())

    classes_reais, classes_previstas, acuracias = executar_validacao_cruzada(X, y)

    matriz = imprimir_matriz_de_confusao(classes_reais, classes_previstas, classes)
    desenhar_matriz_de_confusao(matriz, classes)
    imprimir_relatorio(classes_reais, classes_previstas, acuracias)

    modelo = treinar_modelo_final(X, y)

    importancias = calcular_importancia_dos_atributos(modelo, X)
    imprimir_importancia(importancias)
    desenhar_grafico_de_importancia(importancias)

    classificar_unidades_sem_classe(modelo, list(X.columns))


if __name__ == "__main__":
    classificar()
