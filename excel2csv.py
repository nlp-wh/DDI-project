import openpyxl
import os

data_dir = 'data'

wb = openpyxl.load_workbook(os.path.join(data_dir, 'DDICorpus2013.xlsx'))
ws = wb.active

with open(os.path.join(data_dir, 'ddi.tsv'), 'w', encoding='utf-8') as f:
    for r in ws.rows:
        line = ''
        for cell in r:
            line = line + '{}\t'.format(cell.value)
        line = line.rstrip() + '\n'
        if not line.startswith("None"):  # Check whether it is empty
            f.write(line)
