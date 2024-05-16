import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Data processing')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHC')
args = parser.parse_args()


path = 'result' + args.dataset + '.csv'
data = pd.read_csv(path)
print(path)

df = pd.DataFrame(columns=['tp', 'fp', 'fn', 'tn', 'accuracy', 'f1', 'precision', 'recall', 'FPR', 'Steps'])

for i in range(0,1000,200):
    pre = data['Pre'].to_list()[i:i+200]
    label = data['True'].to_numpy()[i:i+200]
    index = np.where(label == 1)
    steps = data['Steps'].to_numpy()[i:i+200][index]
    print(steps)
    # steps = sum(steps) / len(steps)
    index = np.where(steps < 200)
    steps = sum(steps[index])/ len(steps[index])
    print('steps', steps)

    tn, fp, fn, tp = confusion_matrix(label, pre).ravel()
    accuracy = accuracy_score(label, pre)
    f1 = f1_score(label, pre)
    precision = precision_score(label, pre)
    recall = recall_score(label, pre)
    FPR = fp / (fp + tn)
    f1 = f1_score(label, pre)
    print('-------------' + str(i) + '-------------')
    print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
    print('f1', f1, 'FPR', FPR, 'precision', precision, 'recall', recall, 'accuracy', accuracy, 'steps', steps)
    df.loc[len(df)] = [tp, fp, fn, tn, accuracy, f1, precision, recall, FPR, steps]
print(df)

mean_values = df.mean()

mean_df = pd.DataFrame(mean_values).T  

mean_df.index = ['Average']


result = pd.concat([df, mean_df])
result = result.round(4)
result.to_excel('example.xlsx', index=False,float_format='%.4f', engine='openpyxl')
