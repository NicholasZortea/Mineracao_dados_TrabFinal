"""
Executa todo o pipeline do trabalho, na ordem:

1. uniao_colunas.py  -> work_unido.csv
2. limpeza_dados.py  -> work_padronizado.csv
3. classificador.py  -> resultados no terminal e importancia_atributos.png

Cada etapa tambem pode ser executada sozinha (python <arquivo>.py).
"""

from classificador import classificar
from limpeza_dados import limpar_dados
from uniao_colunas import unir_colunas


def main():
    print("=== Etapa 1: uniao das colunas em comum ===")
    unir_colunas()

    print("\n=== Etapa 2: limpeza dos dados ===")
    limpar_dados()

    print("\n=== Etapa 3: classificacao ===")
    classificar()


if __name__ == "__main__":
    main()
