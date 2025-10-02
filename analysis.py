import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('./data/PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k.csv')
    print(df)
    df[df['Y']==2] = 1
    df = df.drop_duplicates(subset=['X'])
    print(df['Y'])
    print(len(df[df['Y']==1]), len(df[df['Y']==0]))
    # print(df.iloc[569434:]['Y'].mean())
    df.to_csv('./data/PS_revised_labeled_coat+capsid_20221009_labeled_virus_list_decoy_635k_y=1_noseqdup.csv')
