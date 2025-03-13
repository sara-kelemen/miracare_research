import json
import pandas as pd
import numpy as np
import glob

# Load JSON files
with open('data/full_dataset_48days_viterbi (1).json') as fin:
    viterbi = json.load(fin)

with open('data/full_dataset_48days_stochbacktrace (2).json') as fin:
    stoch = json.load(fin)

with open('data/viterbi_alignment_full_dataset_48days_labeled_cycles.json') as fin:
    align = json.load(fin)

# Function to compute alignment scores
def make_avg_pairwise_alignment_col(hubid, output_type):
    if hubid not in align.keys():
        return np.nan
    else:
        scores = list(align[hubid].values())
        if len(scores) == 0:
            print(hubid)
            return np.nan            
        else:
            if output_type == 'mean':
                return np.mean(scores)
            elif output_type == 'min':
                return np.min(scores)
            elif output_type == 'max':
                return np.max(scores)
            elif output_type == 'std':
                return np.std(scores)
            elif output_type == 'median':
                return np.median(scores)

# Function to compute viterbi logprob scores
def make_avg_viterbi_logprob_col(hubid, output_type):
    if hubid not in viterbi.keys():
        return np.nan
    else:
        cycleixs = viterbi[hubid].keys()
        logprobs = [viterbi[hubid][cycleix]['prob'] for cycleix in cycleixs]
        if len(logprobs) == 0:
            print(hubid)
            return np.nan
        else:
            if output_type == 'mean':
                return np.mean(logprobs)
            elif output_type == 'min':
                return np.min(logprobs)
            elif output_type == 'max':
                return np.max(logprobs)
            elif output_type == 'std':
                return np.std(logprobs)
            elif output_type == 'median':
                return np.median(logprobs)

# Function to compute complete logprob scores
def make_avg_complete_logprob_col(hubid, output_type):
    if hubid not in stoch.keys():
        return np.nan
    else:
        cycleixs = stoch[hubid].keys()
        logprobs = [stoch[hubid][cycleix]['logprob'] for cycleix in cycleixs]
        if len(logprobs) == 0:
            print(hubid)
            return np.nan
        else:
            if output_type == 'mean':
                return np.mean(logprobs)
            elif output_type == 'min':
                return np.min(logprobs)
            elif output_type == 'max':
                return np.max(logprobs)
            elif output_type == 'std':
                return np.std(logprobs)
            elif output_type == 'median':
                return np.median(logprobs)

# List of all files to process (replace path with your actual directory)
file_paths = glob.glob('data/hubid_cycle_features_*.csv')

# Process each file
for file_path in file_paths:
    print(f"Processing {file_path}...")
    
    # Read file
    df = pd.read_csv(file_path)

    # Add alignment scores
    for output_type in ['mean', 'min', 'max', 'std', 'median']:
        df[f'viterbi_alignment_score_{output_type}'] = df['hub_id'].apply(lambda x: make_avg_pairwise_alignment_col(x, output_type))
        df[f'viterbi_logprob_{output_type}'] = df['hub_id'].apply(lambda x: make_avg_viterbi_logprob_col(x, output_type))
        df[f'complete_logprob_{output_type}'] = df['hub_id'].apply(lambda x: make_avg_complete_logprob_col(x, output_type))

    # Save updated file
    output_filename = f"cycle_and_HMM_features_{file_path.split('/')[-1]}"
    df.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")