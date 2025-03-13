import pandas as pd
import numpy as np

class Info:

    def __init__(self, path: str = 'data/hub_user_info_export_2024-02-07_110056.csv'):
        self.df = pd.read_csv(path)
    
    def raw_info_data(self):
        return(self.df)
