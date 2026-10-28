"""
Etapa 2 - Limpeza e padronizacao dos dados.

Entrada: work_unido.csv (gerado por uniao_colunas.py).
Saida:   work_padronizado.csv (pronto para o classificador).

Problemas encontrados no dataset e como sao tratados:

1. Espacos extras, inclusive o espaco nao separavel ('\\xa0'), como na Pop do
   Pharaoh (' 0'). -> removidos com strip().
2. Caracteres de anotacao que nao fazem parte do valor:
   '*'  em DmgPierce/DmgCrush (ex.: '11**', '40*')
   '_'  em HP                  (ex.: '1100_')
   '%'  nas colunas de Armor   (ex.: '35%')
   -> removidos.
3. Marcadores de valor vazio: '--', 'none' e celulas em branco.
   - Nas colunas de dano e alcance eles significam "a unidade nao possui esse
     atributo", portanto o valor correto e 0.
   - Em Age, Pop e Train time eles significam "valor desconhecido" (uma idade
     0 ou um tempo de treino 0 nao existem no jogo). Nesses casos o valor e
     preenchido com a MEDIANA da coluna. A justificativa da mediana esta no
     grafico gerado por analise_exploratoria.py.
4. Classes com pouquissimas instancias (Spear Infantary e Ranged Cavalry tem
   apenas 1 unidade cada). Com um unico exemplo o modelo nao aprende a classe
   e ela nao pode ser distribuida entre os folds da validacao cruzada
   estratificada, so atrapalhando as metricas. -> instancias removidas.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = BASE_DIR / "work_unido.csv"
ARQUIVO_SAIDA = BASE_DIR / "work_padronizado.csv"

COLUNA_CLASSE = "Class"

# Textos que representam ausencia de valor no dataset original
MARCADORES_DE_VAZIO = ["--", "none", ""]

# Caracteres de anotacao a remover, por coluna
CARACTERES_A_REMOVER = {
    "DmgPierce": "*",
    "DmgCrush": "*",
    "HP": "_",
    "ArmorHack": "%",
    "ArmorPierce": "%",
    "ArmorCrush": "%",
}

# Colunas em que um valor ausente e desconhecido (e nao "zero")
COLUNAS_PREENCHIDAS_COM_MEDIANA = ["Age", "Pop", "Train time"]

# Classes com menos instancias do que isso sao removidas. O valor acompanha o
# numero de folds da validacao cruzada usada no classificador (4).
MINIMO_DE_INSTANCIAS_POR_CLASSE = 4


def carregar_dados_unidos():
    """Le o CSV gerado pela etapa de uniao, tratando tudo como texto."""
    return pd.read_csv(ARQUIVO_ENTRADA, dtype=str, keep_default_na=False)


def colunas_de_atributos(df):
    """Devolve todas as colunas, exceto a coluna alvo 'Class'."""
    return [coluna for coluna in df.columns if coluna != COLUNA_CLASSE]


def remover_classes_raras(df):
    """Remove as instancias de classes com menos de MINIMO_DE_INSTANCIAS_POR_CLASSE unidades."""
    contagem_por_classe = df[COLUNA_CLASSE].value_counts()
    classes_raras = contagem_por_classe[contagem_por_classe < MINIMO_DE_INSTANCIAS_POR_CLASSE]

    for classe, quantidade in classes_raras.items():
        print(f"[limpeza] Classe '{classe}' removida ({quantidade} instancia(s))")

    df_filtrado = df[~df[COLUNA_CLASSE].isin(classes_raras.index)]
    return df_filtrado.reset_index(drop=True)


def remover_espacos(df):
    """Remove espacos (incluindo '\\xa0') do inicio e do fim de cada valor."""
    df = df.copy()
    for coluna in df.columns:
        df[coluna] = df[coluna].astype(str).str.strip()
    return df


def remover_caracteres_de_anotacao(df):
    """Remove os caracteres '*', '_' e '%' das colunas em que aparecem."""
    df = df.copy()
    for coluna, caractere in CARACTERES_A_REMOVER.items():
        df[coluna] = df[coluna].str.replace(caractere, "", regex=False)
    return df


def converter_para_numerico(df):
    """
    Converte as colunas de atributos para numero.

    Os marcadores de vazio ('--', 'none', '') viram NaN, deixando os valores
    ausentes explicitos para as proximas etapas.
    """
    df = df.copy()
    for coluna in colunas_de_atributos(df):
        coluna_com_nan = df[coluna].replace(MARCADORES_DE_VAZIO, pd.NA)
        df[coluna] = pd.to_numeric(coluna_com_nan, errors="coerce")
    return df


def preparar_dados_com_ausentes():
    """
    Carrega os dados unidos, limpa o texto e converte para numero, mantendo os
    valores ausentes como NaN. Usada tambem pela analise exploratoria, que
    precisa enxergar os dados antes do preenchimento.
    """
    df = carregar_dados_unidos()
    df = remover_espacos(df)
    df = remover_classes_raras(df)
    df = remover_caracteres_de_anotacao(df)
    df = converter_para_numerico(df)
    return df


def preencher_ausentes_com_mediana(df):
    """Preenche Age, Pop e Train time ausentes com a mediana da coluna."""
    df = df.copy()
    for coluna in COLUNAS_PREENCHIDAS_COM_MEDIANA:
        mediana = df[coluna].median()
        quantidade_ausente = df[coluna].isna().sum()

        df[coluna] = df[coluna].fillna(mediana)

        print(f"[limpeza] {coluna}: {quantidade_ausente} ausente(s) preenchido(s) com a mediana ({mediana})")
    return df


def preencher_ausentes_com_zero(df):
    """Nas demais colunas, ausencia significa que o atributo nao existe: vira 0."""
    df = df.copy()
    for coluna in colunas_de_atributos(df):
        if coluna in COLUNAS_PREENCHIDAS_COM_MEDIANA:
            continue
        df[coluna] = df[coluna].fillna(0)
    return df


def converter_colunas_inteiras(df):
    """Converte para int as colunas que so possuem valores inteiros (ex.: 15.0 -> 15)."""
    df = df.copy()
    for coluna in colunas_de_atributos(df):
        valores = df[coluna]
        if (valores % 1 == 0).all():
            df[coluna] = valores.astype(int)
    return df


def limpar_dados():
    """Executa a etapa completa e salva o resultado em work_padronizado.csv."""
    df = preparar_dados_com_ausentes()
    df = preencher_ausentes_com_mediana(df)
    df = preencher_ausentes_com_zero(df)
    df = converter_colunas_inteiras(df)

    df.to_csv(ARQUIVO_SAIDA, index=False)

    print(f"[limpeza] Arquivo salvo em: {ARQUIVO_SAIDA.name}")
    return df


if __name__ == "__main__":
    limpar_dados()
