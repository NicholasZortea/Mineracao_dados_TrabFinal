"""
Executa todo o pipeline do trabalho, na ordem:

1. uniao_colunas.py        -> work_unido.csv
2. limpeza_dados.py        -> work_padronizado.csv
3. analise_exploratoria.py -> analise_exploratoria.png
4. classificador.py        -> resultados no terminal

Cada etapa tambem pode ser executada sozinha (python <arquivo>.py).
"""

from analise_exploratoria import executar_analise
from classificador import classificar
from limpeza_dados import limpar_dados
from uniao_colunas import unir_colunas


def main():
    print("=== Etapa 1: uniao das colunas em comum ===")
    unir_colunas()

    print("\n=== Etapa 2: limpeza dos dados ===")
    limpar_dados()

    print("\n=== Etapa 3: analise exploratoria ===")
    executar_analise()

    print("\n=== Etapa 4: classificacao ===")
    classificar()


if __name__ == "__main__":
    main()
