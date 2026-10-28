import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df_class = pd.read_csv(BASE_DIR / "AgeOfMythology_noClass.csv")
df_work = pd.read_csv(BASE_DIR / "AgeOfMythology_work.csv")


class_column = df_work['Class']

# Extraindo o Favor da coluna Cost 
df_work['Favor'] = pd.to_numeric(df_work['Cost'].str.extract(r'(\d+)\s*Fv', expand=False)).fillna(0).astype(int)

colunas_comum = df_work.columns.intersection(df_class.columns)

df_work = df_work[colunas_comum]
df_work['Class'] = class_column

# Removendo o espaco nao separavel da Pop do Pharaoh ('\xa00' -> '0')
df_work['Pop'] = df_work['Pop'].astype(str).str.strip()


# Removendo os '*' da coluna
df_work['DmgPierce'] = df_work['DmgPierce'].astype(str).str.replace('*', '', regex=False)
df_work['DmgCrush'] = df_work['DmgCrush'].astype(str).str.replace('*', '', regex=False)

df_work['HP'] = df_work['HP'].astype(str).str.replace('_', '', regex=False)

# Preenchendo a Age ausente ('--') com a mediana da coluna, ja que idade 0 nao existe
df_work['Age'] = pd.to_numeric(df_work['Age'], errors='coerce')
df_work['Age'] = df_work['Age'].fillna(df_work['Age'].median()).astype(int)


df_work = df_work.replace('none', 0)
df_work = df_work.replace('--', 0)


colunas_armor = ['ArmorHack', 'ArmorPierce', 'ArmorCrush']

for i in colunas_armor:
    df_work[i] = df_work[i].astype(str).str.replace('%', '', regex=False)
    df_work[i] = pd.to_numeric(df_work[i], errors='coerce').fillna(0)

df_work.to_csv(BASE_DIR / "work_padronizado.csv", index=False)