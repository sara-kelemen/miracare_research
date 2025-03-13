import pandas as pd
from info import Info

class AddPCOS:

    def __init__(self):
        df_user = Info().raw_info_data()
        #if conditions contains 2, then PCOS
        #if conditions contains only 3 or 4, then Baseline
        #otherwise, nonPCOS-nonBaseline
        self.PCOS_ids = df_user.loc[df_user['conditions'].str.contains('2')]['hub_id'].tolist()
        self.Baseline_ids = df_user.loc[(df_user['conditions']=='3') | (df_user['conditions']=='4')]['hub_id'].tolist()

    def add_pcos(self, df):
        '''df a dataframe with one column named hub_id'''
        #df = pd.read_csv(path)
        df['group'] = df['hub_id'].apply(self.mapid_to_group)
        return(df)


    def mapid_to_group(self, id):
        if id in self.PCOS_ids:
            return('PCOS')
        elif id in self.Baseline_ids:
            return('Baseline')
        else:
            return('nonPCOS-nonBaseline')


#testings       
if __name__=='__main__':
    print(AddPCOS().add_pcos(Info().raw_info_data()))