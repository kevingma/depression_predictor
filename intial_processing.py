import pandas as pd
import pyreadstat

# Reads xpt files
def read_xpt(file_path):
    df, meta = pyreadstat.read_xport(file_path)
    return df


files = [
    'P_DEMO.XPT', 'P_BMX.XPT', 'P_DBQ.XPT', 'P_ECQ.XPT', 'P_FSQ.XPT', 'P_HIQ.XPT', 'P_INQ.XPT', 'P_RXQ_RX.XPT',
    'P_DPQ.XPT', 'P_OCQ.XPT', 'P_OHQ.XPT', 'P_RHQ.XPT', 'P_WHQ.XPT', 'P_WHQMEC.XPT', 'P_MCQ.XPT'
]

initial_data = read_xpt('./P_DEMO.XPT')

# Merge XPT files
for file in files[1:]:
    data = read_xpt(f'./{file}')
    initial_data = initial_data.merge(data, on='SEQN', how='left')

# Remove minors
minors_removed = initial_data[initial_data['RIDAGEYR'] >= 18].copy()

#Add column for total depression score
depression_columns = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100']

minors_removed.loc[:, 'depression_score'] = minors_removed[depression_columns].sum(axis=1)

# List of relevant columns
relevant_columns = [
    'SEQN', 'depression_score', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDBORN4',
    'DMDYRUSZ', 'DMDEDUC2', 'DMDMARTZ', 'INDFMPIR', 'BMXWT', 'BMXHT', 'BMXBMI',
    'DBQ010', 'DBQ229', 'DBQ235A', 'DBQ235B', 'DBQ235C', 'DBQ700', 'DBQ301',
    'DBQ330', 'DBQ197', 'DBQ223A', 'DBQ223B', 'DBQ223C', 'DBQ223D', 'DBQ223E',
    'DBQ223U', 'DBD895', 'DBD900', 'DBD905', 'DBD910', 'DBQ940', 'DBQ945',
    'FSDHH', 'FSDAD', 'FSD151', 'FSQ165', 'FSQ012', 'FSD230', 'FSD162', 'FSQ760',
    'FSD652ZW', 'FSD672ZW', 'FSD652CW', 'FSD660ZW', 'DBQ360', 'DBQ370', 'DBD381',
    'DBQ390', 'DBQ400', 'DBD411', 'DBQ421', 'DBQ424', 'HIQ011', 'HIQ032A',
    'HIQ032B', 'HIQ032C', 'HIQ032D', 'HIQ032E', 'HIQ032H', 'HIQ032I', 'HIQ032J',
    'HIQ260', 'HIQ105', 'HIQ270', 'HIQ210', 'INDFMMPI', 'INDFMMPC', 'MCQ010',
    'MCQ025', 'MCQ035', 'MCQ040', 'MCQ050', 'AGQ030', 'MCQ053', 'MCQ080',
    'MCQ151', 'MCQ160A', 'MCQ195', 'MCQ160F', 'MCD180F', 'MCQ160M',
    'MCQ170M', 'MCD180M', 'MCQ520', 'MCQ530', 'MCQ540', 'MCQ550', 'MCQ560',
    'MCQ570', 'MCQ220', 'MCQ230A', 'MCQ230B', 'MCQ230C', 'MCQ230D', 'MCQ366A',
    'MCQ366B', 'MCQ366C', 'MCQ366D', 'MCQ371A', 'MCQ371B', 'MCQ371C', 'MCQ371D',
    'OSQ230', 'OCD150', 'OCQ180', 'OCQ210', 'OCQ670', 'OCD383', 'OHQ030',
    'OHQ033', 'OHQ770', 'OHQ780A', 'OHQ780B', 'OHQ780C', 'OHQ780D', 'OHQ780E',
    'OHQ780F', 'OHQ780G', 'OHQ780H', 'OHQ780I', 'OHQ780J', 'OHQ780K', 'OHQ620',
    'OHQ640', 'OHQ835', 'RXDUSE', 'RXDDRUG', 'RXDDRGID', 'RXQSEEN', 'RXDDAYS',
    'RXDRSC1', 'RXDRSC2', 'RXDRSC3', 'RXDRSD1', 'RXDRSD2', 'RXDRSD3', 'RXDCOUNT',
    'RHQ074', 'RHQ076', 'RHQ131', 'RHD143', 'RHQ160', 'RHD167', 'RHQ171',
    'RHQ197', 'RHQ200', 'WHD010', 'WHD020', 'WHQ030', 'WHQ040', 'WHD050',
    'WHQ060', 'WHQ070', 'WHD140', 'WHQ030M', 'WHQ500', 'WHQ520'
]

# Remove irrelevant questions
filtered_data = minors_removed[relevant_columns]

# Save
filtered_data.to_csv('processed_data.csv', index=False)


