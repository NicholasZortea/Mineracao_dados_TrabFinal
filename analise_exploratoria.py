"""
Etapa 3 - Analise exploratoria.

Gera uma unica figura (analise_exploratoria.png) com um boxplot para cada
coluna numerica, calculado ANTES do preenchimento dos valores ausentes.
Cada boxplot mostra:
- a caixa, do 1o ao 3o quartil (os 50% centrais dos dados);
- a linha azul dentro da caixa, que e a MEDIANA;
- o circulo laranja, que e a MEDIA;
- as hastes, ate o ultimo valor dentro de 1.5 x o intervalo interquartil;
- os pontos isolados, que sao os valores extremos (outliers).

Como as colunas tem escalas muito diferentes (HP chega a 1200 e Speed fica
perto de 4), cada boxplot tem o seu proprio eixo, com os valores reais.

Leitura do grafico para justificar a mediana:
- Quando a media (circulo) se afasta da mediana (linha), a coluna e
  assimetrica ou tem valores extremos que "puxam" a media.
- A mediana nao e afetada por esses extremos e, nas colunas discretas como
  Age e Pop, ela sempre e um valor que realmente existe no jogo (a media
  geraria, por exemplo, uma idade 2.75).
- Nas colunas de dano e alcance, os vazios significam "a unidade nao tem esse
  tipo de dano"; por isso eles viram 0 e as metricas consideram so quem tem.
Por isso a mediana e a escolha mais segura para preencher os ausentes.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from limpeza_dados import (
    COLUNAS_PREENCHIDAS_COM_MEDIANA,
    colunas_de_atributos,
    preparar_dados_com_ausentes,
)


BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_GRAFICO = BASE_DIR / "analise_exploratoria.png"

COLUNAS_POR_LINHA_DA_FIGURA = 5

COR_MEDIANA = "#2a78d6"
COR_MEDIA = "#eb6834"
COR_CAIXA = "#e4e3dd"
COR_CAIXA_DESTAQUE = "#d4e4f7"
COR_BORDA = "#8a8983"
COR_GRADE = "#e4e3dd"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_FUNDO = "#fcfcfb"


def calcular_metricas(df):
    """Calcula as metricas descritivas de cada coluna de atributo."""
    linhas = []
    for coluna in colunas_de_atributos(df):
        valores = df[coluna]
        linhas.append({
            "coluna": coluna,
            "minimo": valores.min(),
            "q1": valores.quantile(0.25),
            "mediana": valores.median(),
            "media": valores.mean(),
            "q3": valores.quantile(0.75),
            "maximo": valores.max(),
            "desvio_padrao": valores.std(),
            "assimetria": valores.skew(),
            "ausentes": int(valores.isna().sum()),
        })
    return pd.DataFrame(linhas).set_index("coluna")


def imprimir_metricas(metricas):
    """Mostra a tabela de metricas no terminal."""
    print("[analise] Metricas descritivas (antes do preenchimento dos ausentes):")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.2f}".format):
        print(metricas)


def ordenar_colunas_para_o_grafico(df):
    """Coloca primeiro as colunas preenchidas com a mediana, depois as demais."""
    colunas = colunas_de_atributos(df)
    colunas_em_destaque = [c for c in colunas if c in COLUNAS_PREENCHIDAS_COM_MEDIANA]
    demais_colunas = [c for c in colunas if c not in COLUNAS_PREENCHIDAS_COM_MEDIANA]
    return colunas_em_destaque + demais_colunas


def descrever_ausentes(coluna, quantidade_ausente):
    """Texto curto dizendo quantos valores faltavam e como foram preenchidos."""
    if quantidade_ausente == 0:
        return "sem ausentes"
    if coluna in COLUNAS_PREENCHIDAS_COM_MEDIANA:
        return f"{quantidade_ausente} ausente(s) -> mediana"
    return f"{quantidade_ausente} sem valor -> 0"


def desenhar_boxplot_da_coluna(eixo, coluna, valores, metricas_da_coluna):
    """Desenha o boxplot de uma coluna dentro do eixo informado."""
    em_destaque = coluna in COLUNAS_PREENCHIDAS_COM_MEDIANA
    cor_da_caixa = COR_CAIXA_DESTAQUE if em_destaque else COR_CAIXA

    eixo.boxplot(
        valores.dropna(),
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        boxprops={"facecolor": cor_da_caixa, "edgecolor": COR_BORDA, "linewidth": 1},
        whiskerprops={"color": COR_BORDA, "linewidth": 1},
        capprops={"color": COR_BORDA, "linewidth": 1},
        medianprops={"color": COR_MEDIANA, "linewidth": 2.5},
        meanprops={"marker": "o", "markerfacecolor": COR_MEDIA,
                   "markeredgecolor": COR_FUNDO, "markersize": 8},
        flierprops={"marker": "o", "markerfacecolor": "none",
                    "markeredgecolor": COR_BORDA, "markersize": 4},
    )

    eixo.set_title(coluna, fontsize=11, color=COR_TEXTO, loc="left",
                   fontweight="bold" if em_destaque else "normal")

    # Resumo com os valores reais embaixo de cada boxplot
    resumo = (f"mediana {metricas_da_coluna['mediana']:g}  |  "
              f"media {metricas_da_coluna['media']:.2f}\n"
              f"{descrever_ausentes(coluna, int(metricas_da_coluna['ausentes']))}")
    eixo.text(0.0, -0.08, resumo, transform=eixo.transAxes, va="top",
              fontsize=8.5, color=COR_TEXTO_SECUNDARIO)

    eixo.set_facecolor(COR_FUNDO)
    eixo.set_xticks([])
    eixo.tick_params(axis="y", labelsize=8, colors=COR_TEXTO_SECUNDARIO, length=0)
    eixo.grid(axis="y", color=COR_GRADE, linewidth=0.6)
    eixo.set_axisbelow(True)
    for lado in ["top", "right", "bottom"]:
        eixo.spines[lado].set_visible(False)
    eixo.spines["left"].set_color(COR_GRADE)


def adicionar_legenda(figura):
    """Legenda unica, no rodape da figura."""
    itens = [
        Line2D([], [], color=COR_MEDIANA, linewidth=2.5, label="Mediana"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COR_MEDIA,
               markeredgecolor=COR_FUNDO, markersize=8, label="Media"),
        Patch(facecolor=COR_CAIXA, edgecolor=COR_BORDA, label="Q1 a Q3 (50% central)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=COR_BORDA, markersize=5, label="Valores extremos"),
        Patch(facecolor=COR_CAIXA_DESTAQUE, edgecolor=COR_BORDA,
              label="Coluna preenchida com a mediana"),
    ]
    figura.legend(handles=itens, loc="lower center", ncol=len(itens),
                  frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))


def desenhar_grafico(df, metricas):
    """Monta a figura com um boxplot por coluna e salva em PNG."""
    colunas = ordenar_colunas_para_o_grafico(df)
    quantidade_de_linhas = math.ceil(len(colunas) / COLUNAS_POR_LINHA_DA_FIGURA)

    figura, eixos = plt.subplots(
        quantidade_de_linhas, COLUNAS_POR_LINHA_DA_FIGURA,
        figsize=(15, 4 * quantidade_de_linhas),
    )
    figura.patch.set_facecolor(COR_FUNDO)
    eixos = eixos.flatten()

    for eixo, coluna in zip(eixos, colunas):
        desenhar_boxplot_da_coluna(eixo, coluna, df[coluna], metricas.loc[coluna])

    # Esconde os espacos que sobraram na grade
    for eixo in eixos[len(colunas):]:
        eixo.set_visible(False)

    figura.suptitle(
        "Distribuicao de cada atributo antes do preenchimento: nas colunas assimetricas a media "
        "e puxada pelos extremos,\nenquanto a mediana se mantem estavel e e sempre um valor real "
        "do jogo - por isso ela preenche os ausentes",
        fontsize=13, color=COR_TEXTO, x=0.01, ha="left",
    )
    adicionar_legenda(figura)

    figura.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=4.5)
    figura.savefig(ARQUIVO_GRAFICO, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)
    plt.close(figura)


def executar_analise():
    """Executa a etapa completa: calcula, imprime e salva o grafico."""
    df = preparar_dados_com_ausentes()
    metricas = calcular_metricas(df)

    imprimir_metricas(metricas)
    desenhar_grafico(df, metricas)

    print(f"[analise] Grafico salvo em: {ARQUIVO_GRAFICO.name}")
    return metricas


if __name__ == "__main__":
    executar_analise()
