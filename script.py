import pandas as pd

df_class = pd.read_csv("/home/visitante/dm/AgeOfMythology_noClass.csv")
df_work = pd.read_csv("/home/visitante/dm/AgeOfMythology_work.csv")


class_column = df_work['Class']

colunas_comum = df_work.columns.intersection(df_class.columns)

df_work = df_work[colunas_comum]
df_work['Class'] = class_column

df_work.to_csv("work_padronizado.csv", index=False)


coluna_teste = df_work['DmgPierce']
lista_limpa = coluna_teste.replace('--', pd.NA).dropna()
print(lista_limpa.describe.mean())
