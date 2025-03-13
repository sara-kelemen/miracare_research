#Now doing this with ALL baseline and ALL PCOS. Moved previous to _OLD
import json
import pandas as pd
import numpy as np

#for alignment scores (averaged over all pairwise comparisons):
with open('processed/viterbi_alignment_full_dataset_48days.json') as fin:
    align = json.load(fin)

#for logprobs:
with open('processed/full_dataset_48days_viterbi.json') as fin:
    viterbi = json.load(fin)

with open('processed/full_dataset_48days_stochbacktrace.json') as fin:
    stoch = json.load(fin)

#cycle features file: (want to add other features)
df = pd.read_csv('processed/hubid_cycle_features.csv')

def make_avg_pairwise_alignment_col(hubid, output_type):
    if hubid not in align.keys():
        return np.nan
    else:
        scores = align[hubid]
        if len(scores)==0:
            print(hubid)
            return(np.nan)            
        else:
            if output_type=='mean':
                return(np.mean(scores))
            elif output_type=='min':
                return(np.min(scores))
            elif output_type=='max':
                return(np.max(scores))
            elif output_type=='std':
                return(np.std(scores))
            elif output_type=='median':
                return(np.median(scores))


def make_avg_viterbi_logprob_col(hubid, output_type):
    if hubid not in viterbi.keys():
        return(np.nan)
    else:
        cycleixs = viterbi[hubid].keys()
        logprobs = []
        for cycleix in cycleixs:
            logprobs.append(viterbi[hubid][cycleix]['prob'])
        if len(logprobs)==0:
            print(hubid)
            return(np.nan)
        else:
            if output_type=='mean':
                return(np.mean(logprobs))
            elif output_type=='min':
                return(np.min(logprobs))
            elif output_type=='max':
                return(np.max(logprobs))
            elif output_type=='std':
                return(np.std(logprobs))
            elif output_type=='median':
                return(np.median(logprobs))

def make_avg_complete_logprob_col(hubid, output_type):
    if hubid not in stoch.keys():
        return(np.nan)
    else:
        cycleixs = stoch[hubid].keys()
        logprobs = []
        for cycleix in cycleixs:
            logprobs.append(stoch[hubid][cycleix]['logprob'])
        if len(logprobs)==0:
            print(hubid)
            return(np.nan)
        else:
            if output_type=='mean':
                return(np.mean(logprobs))
            elif output_type=='min':
                return(np.min(logprobs))
            elif output_type=='max':
                return(np.max(logprobs))
            elif output_type=='std':
                return(np.std(logprobs))
            elif output_type=='median':
                return(np.median(logprobs))

#add cycle alignment features
for output_type in ['mean','min','max','std','median']:
    df['viterbi_alignment_score_'+output_type] = df['hub_id'].apply(lambda x: make_avg_pairwise_alignment_col(x, output_type))

for output_type in ['mean','min','max','std','median']:
    df['viterbi_logprob_'+output_type] = df['hub_id'].apply(lambda x: make_avg_viterbi_logprob_col(x, output_type))

for output_type in ['mean','min','max','std','median']:
    df['complete_logprob_'+output_type] = df['hub_id'].apply(lambda x: make_avg_complete_logprob_col(x, output_type))


#df = pd.merge(df1, df2, how='outer')
#df = pd.merge(df, df3, on='hub_id', how='left')
df.to_csv('processed/cycle_and_HMM_features_full_dataset_48days.csv', index=False)
