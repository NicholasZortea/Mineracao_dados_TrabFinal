"""
Etapa 1 - Uniao das colunas presentes em ambos os datasets.

O dataset de trabalho (AgeOfMythology_work.csv) possui mais colunas do que o
dataset sem classe (AgeOfMythology_noClass.csv). Para que o classificador
treinado com o primeiro possa ser aplicado ao segundo, mantemos no dataset de
trabalho apenas as colunas que existem nos dois arquivos, mais a coluna alvo
'Class'.

A coluna 'Favor' nao existe diretamente no dataset de trabalho: ela esta
embutida no texto da coluna 'Cost' (ex.: "250F 22Fv" -> 22 de Favor). Por isso
ela e extraida antes da intersecao das colunas.

Saida: work_unido.csv (ainda com os valores "sujos", sem limpeza).
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_TRABALHO = BASE_DIR / "AgeOfMythology_work.csv"
ARQUIVO_SEM_CLASSE = BASE_DIR / "AgeOfMythology_noClass.csv"
ARQUIVO_SAIDA = BASE_DIR / "work_unido.csv"

COLUNA_CLASSE = "Class"
COLUNA_CUSTO = "Cost"
COLUNA_FAVOR = "Favor"


def carregar_datasets():
    """Le os dois arquivos CSV originais e devolve (df_trabalho, df_sem_classe)."""
    df_trabalho = pd.read_csv(ARQUIVO_TRABALHO)
    df_sem_classe = pd.read_csv(ARQUIVO_SEM_CLASSE)
    return df_trabalho, df_sem_classe


def extrair_favor_do_custo(df_trabalho):
    """
    Cria a coluna 'Favor' a partir do texto da coluna 'Cost'.

    O Favor aparece como um numero seguido de "Fv" (ex.: "300G 30Fv").
    Unidades cujo custo nao menciona Favor recebem 0, pois de fato nao gastam
    Favor para serem treinadas.
    """
    df = df_trabalho.copy()

    favor_em_texto = df[COLUNA_CUSTO].str.extract(r"(\d+)\s*Fv", expand=False)
    favor_numerico = pd.to_numeric(favor_em_texto)

    df[COLUNA_FAVOR] = favor_numerico.fillna(0).astype(int)
    return df


def manter_colunas_em_comum(df_trabalho, df_sem_classe):
    """
    Mantem no dataset de trabalho somente as colunas que tambem existem no
    dataset sem classe, preservando a ordem do dataset sem classe e
    acrescentando a coluna 'Class' no final.
    """
    colunas_em_comum = [
        coluna for coluna in df_sem_classe.columns
        if coluna in df_trabalho.columns
    ]

    colunas_finais = colunas_em_comum + [COLUNA_CLASSE]
    return df_trabalho[colunas_finais].copy()


def unir_colunas():
    """Executa a etapa completa e salva o resultado em work_unido.csv."""
    df_trabalho, df_sem_classe = carregar_datasets()

    df_trabalho = extrair_favor_do_custo(df_trabalho)
    df_unido = manter_colunas_em_comum(df_trabalho, df_sem_classe)

    df_unido.to_csv(ARQUIVO_SAIDA, index=False)

    print(f"[uniao] {len(df_unido.columns)} colunas mantidas: {list(df_unido.columns)}")
    print(f"[uniao] Arquivo salvo em: {ARQUIVO_SAIDA.name}")
    return df_unido


if __name__ == "__main__":
    unir_colunas()
