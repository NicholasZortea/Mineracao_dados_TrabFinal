import pandas as pd

df_class = pd.read_csv("/home/leonardowd/dm/Mineracao_dados_TrabFinal/AgeOfMythology_noClass.csv", encoding='utf-8-sig')
df_work = pd.read_csv("/home/leonardowd/dm/Mineracao_dados_TrabFinal/AgeOfMythology_work.csv", encoding='utf-8-sig')


class_column = df_work['Class']

# Extraindo o Favor da coluna Cost 
df_work['Favor'] = df_work['Cost'].str.extract(r'(\d+)\s*Fv', expand=False).fillna(0).astype(int)

colunas_comum = df_work.columns.intersection(df_class.columns)

df_work = df_work[colunas_comum]
df_work['Class'] = class_column

# Removendo o espaco nao separavel da Pop do Pharaoh ('\xa00' -> '0')
df_work['Pop'] = df_work['Pop'].astype(str).str.strip()

# Juntando classes com apenas 1 exemplo a classes parecidas
df_work['Class'] = df_work['Class'].replace({
    'Spear Infantary': 'Melee Infantary',
    'Ranged Cavalry': 'Cavalry',
})

# Removendo os '*' da coluna
df_work['DmgPierce'] = df_work['DmgPierce'].astype(str).str.replace('*', '', regex=False)
df_work['DmgCrush'] = df_work['DmgCrush'].astype(str).str.replace('*', '', regex=False)

df_work['HP'] = df_work['HP'].astype(str).str.replace('_', '', regex=False)



df_work = df_work.replace('none', 0)
df_work = df_work.replace('--', 0)


colunas_armor = ['ArmorHack', 'ArmorPierce', 'ArmorCrush']

for i in colunas_armor:
    df_work[i] = df_work[i].astype(str).str.replace('%', '', regex=False)
    df_work[i] = pd.to_numeric(df_work[i], errors='coerce').fillna(0)

df_work.to_csv("work_padronizado.csv", index=False)