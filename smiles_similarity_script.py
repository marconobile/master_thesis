#!/home/marcon/miniconda3/bin/python

from rdkit import Chem
from thefuzz import fuzz
import os
import csv
import pickle

my_files = ['path_to_generated_smiles']


data_files = []
path = 'path_to_csvs_in_pickle_format'
for root, dirs, files in os.walk(path, topdown=False):
    for file in files:
        data_files.append(path + '/' + file)


cwd = os.getcwd()

for epoch_, txt in enumerate(my_files):
    for file in data_files:

        with open(file, 'rb') as f:
            if file[file.rfind('/')+1] == '.':
                continue
            else:
                print(file)
                suppl_asinex_smiles = pickle.load(f)

        txt_report_file = txt
        smiles_ = []

        with open(txt_report_file, 'r') as f:
            for i, line in enumerate(f):
                beginning_of_smile = line.find(":") + 1
                end_of_smile = line[beginning_of_smile:].find(',')
                smi = str(line[beginning_of_smile:][:end_of_smile]).strip()
                smiles_.append(smi)
        smiles_.insert(0, 'na')

        with open(f'/{file}_{epoch_}.csv', 'w') as f:
            writer = csv.writer(f)

            # first row of table: my generated smiles
            writer.writerow(smiles_)

            for idx_data_smi, data_smi in enumerate(suppl_asinex_smiles):  # for each row
                row = [data_smi]
                for idx_gen_smi, gen_smi in enumerate(smiles_):  # for each col
                    if idx_gen_smi == 0:  # if col =0
                        continue  # skip it

                    row.append(fuzz.ratio(data_smi, gen_smi))
                writer.writerow(row)
