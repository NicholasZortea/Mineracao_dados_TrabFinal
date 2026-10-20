import pandas as pd

df_class = pd.read_csv("/home/visitante/dm/AgeOfMythology_noClass.csv")
df_work = pd.read_csv("/home/visitante/dm/AgeOfMythology_work.csv")


class_column = df_work['Class']

colunas_comum = df_work.columns.intersection(df_class.columns)

df_work = df_work[colunas_comum]
df_work['Class'] = class_column

# Removendo os '*' da coluna
df_work['DmgPierce'] = df_work['DmgPierce'].astype(str).str.replace('*', '', regex=False)


df_work = df_work.replace('--', 0)

colunas_armor = ['ArmorHack', 'ArmorPierce', 'ArmorCrush']

for i in colunas_armor:
    df_work[i] = df_work[i].astype(str).str.replace('%', '', regex=False)
    df_work[i] = pd.to_numeric(df_work[i], errors='coerce').fillna(0)
    df_work[i] = df_work[i] / 100

print(df_work)