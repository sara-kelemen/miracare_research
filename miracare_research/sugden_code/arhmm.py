'''AR-HMM code'''
import pandas as pd
import numpy as np
import scipy

class ARHMM:

    def __init__(self, regression_param_file):
        #import regression params
        self.df_regression_params = pd.read_csv(regression_param_file)
        #day 0 emission params
        self.day0_lh_scale=4.48
        self.day0_pdg_scale = 8.15
        self.day0_e3g_scale = 116.3
        self.Transition = np.matrix([[0,1,0,0,0,0,0,0],[0,0.833,0.167,0,0,0,0,0],[0,0,0.888,0.112,0,0,0,0],[0,0,0,0.667,0.333,0,0,0],
                       [0,0,0,0,0.8,0.2,0,0],[0,0,0,0,0,0.8,0.2,0],[0,0,0,0,0,0,0.833,0.167]])


#    def transition_prob(self, a,b):
#        if a == 0 and b == 1:
#            return(1)
#        if a == 6 and b == 7:
#            return(1)
#        elif (a == b) and (a!= 0) and (a!= 7):
#            return(0.8)
#        elif b - a == 1:
#            return(0.2)
#        else:
#            return(0)
    
    def transition_prob(self, a, b):
        a = int(a)
        b = int(b)
        transitions=self.Transition
        #T = np.matrix([[0,1,0,0,0,0,0,0],[0,0.833,0.167,0,0,0,0,0],[0,0,0.888,0.112,0,0,0,0],[0,0,0,0.667,0.333,0,0,0],
        #               [0,0,0,0,0.8,0.2,0,0],[0,0,0,0,0,0.8,0.2,0],[0,0,0,0,0,0,0.833,0.167]])
        #print(a,b)
        return(transitions[a, b])
    
    def emission(self, h, p, pv, cv):
        emissions = self.df_regression_params
        slope = emissions['Slope'].loc[(emissions['Phase']==p) & (emissions['Hormone']==h)].iloc[0]
        intercept = emissions['Intercept'].loc[(emissions['Phase']==p) & (emissions['Hormone']==h)].iloc[0]
        std_dev = emissions['Std Dev'].loc[(emissions['Phase']==p) & (emissions['Hormone']==h)].iloc[0]
        return scipy.stats.norm.pdf(cv, intercept + pv*slope, std_dev)
    
    def day0_emissions(self, lh_val, pdg_val, e3g_val, emission_type):
        if emission_type=='all':
            return(scipy.stats.expon.pdf(lh_val, scale = self.day0_lh_scale) * scipy.stats.expon.pdf(pdg_val, scale = self.day0_pdg_scale) * scipy.stats.expon.pdf(e3g_val, scale = self.day0_e3g_scale))
        elif emission_type=='LH':
            return(scipy.stats.expon.pdf(lh_val, scale = self.day0_lh_scale)) 
        elif emission_type=='PDG':
            return(scipy.stats.expon.pdf(pdg_val, scale = self.day0_pdg_scale))
        elif emission_type=='E3G':
            return(scipy.stats.expon.pdf(e3g_val, scale=self.day0_e3g_scale))
    
    def logsumexp(self, logvec):
        '''take in log values in an array
        output log(sum(exp(log_values)))'''
        m = max(logvec)
        if m == -np.inf:
            return(-np.inf)
        else:
            return(m+np.log(sum(np.exp(np.array(logvec)-m))))

    def backward(self, lh_sequence, pdg_sequence, e3g_sequence, emission_type='all'):
        L = len(lh_sequence)
        # pad hormone sequences to make indexes align
        lh_sequence = [np.nan]+lh_sequence+[np.nan]
        pdg_sequence = [np.nan]+pdg_sequence+[np.nan]
        e3g_sequence = [np.nan]+e3g_sequence+[np.nan]      
        #logb  is a (6+2)x(L+2) matrix holding P(all data after position i and in state 'phase' at position i)
        logb = np.zeros(shape=(8, L+2))
        logb.fill(-np.inf)    

        #must start in state 7 at position L+1
        logb[7, L+1] = 0 #probability=1

        #day L: only relevant phase is 6
        logb[6, L] = np.log(self.transition_prob(6,7))

        #day L-1 down to day 1
        for i in range(L-1,0,-1):
            #hormone values at day i
            lhval = lh_sequence[i+1]
            pdgval = pdg_sequence[i+1]
            e3gval = e3g_sequence[i+1]
            lhval_prev = lh_sequence[i]
            pdgval_prev = pdg_sequence[i]
            e3gval_prev = e3g_sequence[i]  

            for phase in range(1,7):          
                vec_to_sum = []
                for l in range(1,7):
                    emit_lh = self.emission('LH', l, lhval_prev, lhval)
                    emit_pdg = self.emission('PDG', l, pdgval_prev, pdgval)
                    emit_e3g = self.emission('E3G', l, e3gval_prev, e3gval)
                    if emission_type=='all':
                        log_emit = np.log(emit_lh) + np.log(emit_pdg) + np.log(emit_e3g)
                    elif emission_type=='LH':
                        log_emit = np.log(emit_lh)
                    elif emission_type=='PDG':
                        log_emit = np.log(emit_pdg)
                    elif emission_type=='E3G':
                        log_emit = np.log(emit_e3g)
                    transit = self.transition_prob(phase, l)
                    logback = logb[l, i+1]
                    if (transit==0) or (logback==-np.inf):
                        vec_to_sum.append(-np.inf)
                    else:
                        vec_to_sum.append(np.log(transit) + log_emit + logback)
                logb[phase, i] = self.logsumexp(vec_to_sum)
        #finish: i=0
        #first day emissions:
        lhval = lh_sequence[1]
        pdgval = pdg_sequence[1]
        e3gval = e3g_sequence[1]
        emit = self.day0_emissions(lhval, pdgval, e3gval, emission_type=emission_type)
        #only relevant phase at 0 is 0, and at 1 is 1
        logb[0,0] = np.log(self.transition_prob(0,1)) + np.log(emit) + logb[1,1]
        logprob_data = logb[0,0]
        return({'logb':logb, 'logprob':logprob_data})        
                         


    def forward(self, lh_sequence, pdg_sequence, e3g_sequence, emission_type='all'):
        L = len(lh_sequence)
        # pad hormone sequences to make indexes align
        lh_sequence = [np.nan]+lh_sequence+[np.nan]
        pdg_sequence = [np.nan]+pdg_sequence+[np.nan]
        e3g_sequence = [np.nan]+e3g_sequence+[np.nan]

        #logf  is a (6+2)x(L+2) matrix holding P(all data up to position i and in state 'phase' at position i)
        logf = np.zeros(shape=(8, L+2))
        logf.fill(-np.inf)    

        #must start in state 0 at position 0 (prob=1)
        logf[0,0] = 0

        #first day emissions:
        lhval = lh_sequence[1]
        pdgval = pdg_sequence[1]
        e3gval = e3g_sequence[1]
        emit = self.day0_emissions(lhval, pdgval, e3gval, emission_type=emission_type)
        #has to go to phase 1
        vec_to_sum = np.array([logf[k,0] + np.log(self.transition_prob(k,1))
                               if self.transition_prob(k,1)!=0 else -np.inf for k in range(7)])
        #need sum of exp(items), then take log
        #forward update, only relevant phase is phase 1 at position 1
        logf[1,1] = self.logsumexp(vec_to_sum) + np.log(emit)
        #everything else is still -inf    
        
        #iterate over rest of sequence
        for i in range(2, L+1):
            #hormone values at day i
            lhval = lh_sequence[i]
            pdgval = pdg_sequence[i]
            e3gval = e3g_sequence[i]
            lhval_prev = lh_sequence[i-1]
            pdgval_prev = pdg_sequence[i-1]
            e3gval_prev = e3g_sequence[i-1]
            #iterate over potential phases at day i
            for phase in range(1,7):
                #emission
                emit_lh = self.emission('LH', phase, lhval_prev, lhval)
                emit_pdg = self.emission('PDG', phase, pdgval_prev, pdgval)
                emit_e3g = self.emission('E3G', phase, e3gval_prev, e3gval)
                if emission_type=='all':
                    #print(emit_lh, emit_pdg, emit_e3g)
                    log_emit = np.log(emit_lh) + np.log(emit_pdg) + np.log(emit_e3g)
                elif emission_type=='LH':
                    log_emit = np.log(emit_lh)
                elif emission_type=='PDG':
                    log_emit = np.log(emit_pdg)
                elif emission_type=='E3G':
                    log_emit = np.log(emit_e3g)
                #vector of ways to get to day i
                vec_to_sum = np.array([logf[k,i-1] + np.log(self.transition_prob(k, phase))
                                       if self.transition_prob(k, phase)!=0 else -np.inf for k in range(7)])
                #forward update
                logf[phase, i] = log_emit + self.logsumexp(vec_to_sum)

        #finish (L+1 must be in phase 7)
        vec_to_sum = np.array([logf[k, L] + np.log(self.transition_prob(k, 7))
                               if self.transition_prob(k,7)!=0 else -np.inf for k in range(7)])
        logprob_data = self.logsumexp(vec_to_sum)
        logf[7, L+1] = logprob_data
        #print(logprob_data)
        return({'logf':logf, 'logprob':logprob_data})    

    def gamma_probabilities(self, lhvals, pdgvals, e3gvals, emission_type='all'):
        L = len(lhvals)
        F = self.forward(lh_sequence=lhvals, pdg_sequence=pdgvals, e3g_sequence=e3gvals, emission_type=emission_type)
        B = self.backward(lh_sequence=lhvals, pdg_sequence=pdgvals, e3g_sequence=e3gvals, emission_type=emission_type)
        if abs(F['logprob']-B['logprob'])>0.001:
            print('forward/backward mismatch')
        else:
            logf = F['logf']
            logb = B['logb']
            #each is (6+2)x(L+2)
            gamma = np.zeros((8, L+2))
            for i in range(L+2):
                log_vec_to_sum = []
                for phase in range(8):
                    log_vec_to_sum.append(logf[phase, i] + logb[phase, i])
                logsum = self.logsumexp(log_vec_to_sum)
                for phase in range(8):
                    gamma[phase, i] = np.exp(log_vec_to_sum[phase]-logsum)
        return(gamma)
                    

    def multiple_backtraces_and_viterbi(self, outfilename, lhvals, pdgvals, e3gvals, numsamples=100, emission_type='all'):
        N = len(lhvals)
        #first run forward algorithm and viterbi algorithm
        v = self.viterbi(lh_sequence=lhvals, pdg_sequence=pdgvals, e3g_sequence=e3gvals, emission_type=emission_type)['path']
        v = v[1:(len(v)-1)]
        logf = self.forward(lh_sequence=lhvals, pdg_sequence=pdgvals, e3g_sequence=e3gvals, emission_type=emission_type)
        print('logprob of data:'+str(logf['logprob']))
        gamma = self.gamma_probabilities(lhvals, pdgvals, e3gvals, emission_type=emission_type)
        #gamma = gamma[1:(len(gamma)-1)]

        sbs = []
        for i in range(numsamples):
            sb = self.single_stoch_backtrace(logf['logf'])
            sb = sb[1:(len(sb)-1)]
            sbs.append(sb)
        D = {'day':range(N), 'LH':lhvals, 'PDG':pdgvals, 'E3G':e3gvals, 'viterbi':v}
        for phase in range(1,7):
            D['P(phase '+str(phase)+')'] = [gamma[phase, j] for j in range(1,N+1)]
        for i in range(numsamples):
            D['sb'+str(i)] = sbs[i]
        D = pd.DataFrame(D)
        D.to_csv('output/'+outfilename+'.csv', index=False)

    def single_stoch_backtrace(self, logf):
        '''logf is output of forward algorithm'''
        #has to end in state 7 at L+1
        L = logf.shape[1]-2
        path = np.zeros(L+2)
        path[L+1] = 7
        proper_path=True
        for i in range(L, -1, -1):
            if proper_path:
                #print(path[i+1])
                probs = [self.transition_prob(k, path[i+1])*np.exp(logf[k, i]) for k in range(7)]
                sumprobs = sum(probs)
                if sumprobs==0:
                    proper_path = False
                else:
                    probs_normalized = [float(x)/sumprobs for x in probs]
                    #choose path[i] in proportion to these
                    path[i] = np.random.choice(range(7), size=1, p=probs_normalized)[0]
            else: 
                path.fill(np.nan)
                return(path)
        #print(path)
        #print(len(path))
        return(path)


    def viterbi(self, lh_sequence, pdg_sequence, e3g_sequence, emission_type='all'):
        ''' lh_sequence, pdg_sequence, e3g_sequence are lists'''
        ''' emission_type='all' combines all hormones. Or can just use one of LH, PDG, E3G'''
        L = len(lh_sequence) #length of sequences/length of cycle
        # pad hormone sequences to make indexes align
        lh_sequence = [np.nan]+lh_sequence+[np.nan]
        pdg_sequence = [np.nan]+pdg_sequence+[np.nan]
        e3g_sequence = [np.nan]+e3g_sequence+[np.nan]
        # logv is a (6+2)x(L+2) matrix (begin and end states + 6 phases, begin and end locs + N emissions)
        logv = np.zeros(shape=(8, L+2))
        logv.fill(-np.inf)
        #pointer p has the same shape as v
        ptr = np.zeros(shape=(8, L+2))

        logv[0,0] = 0

        #treat first day separately -- no autoregression
        lhval = lh_sequence[1]
        pdgval = pdg_sequence[1]
        e3gval = e3g_sequence[1]
        emit = self.day0_emissions(lhval, pdgval, e3gval, emission_type=emission_type)
        #has to go to phase 1
        vec_to_max = np.array([logv[k,0] + np.log(self.transition_prob(k,1))
                               if self.transition_prob(k,1)!=0 else -np.inf for k in range(7)])
        maxval = vec_to_max.max()
        #pointer from phase 1 at day 1
        ptr[1,1] = vec_to_max.argmax()
        #viterbi update
        logv[1, 1] = np.log(emit)+maxval
        #everything else is still -inf


        for i in range(2, L+1):
            #hormone values at day i
            lhval = lh_sequence[i]
            pdgval = pdg_sequence[i]
            e3gval = e3g_sequence[i]
            lhval_prev = lh_sequence[i-1]
            pdgval_prev = pdg_sequence[i-1]
            e3gval_prev = e3g_sequence[i-1]
            #iterate over potential phases at day i
            for phase in range(1,7):
                #emission
                emit_lh = self.emission('LH', phase, lhval_prev, lhval)
                emit_pdg = self.emission('PDG', phase, pdgval_prev, pdgval)
                emit_e3g = self.emission('E3G', phase, e3gval_prev, e3gval)
                if emission_type=='all':
                    log_emit = np.log(emit_lh) + np.log(emit_pdg) + np.log(emit_e3g)
                elif emission_type=='LH':
                    log_emit = np.log(emit_lh)
                elif emission_type=='PDG':
                    log_emit = np.log(emit_pdg)
                elif emission_type=='E3G':
                    log_emit = np.log(emit_e3g)
                #vector of ways to get to day i
                vec_to_max = np.array([logv[k,i-1] + np.log(self.transition_prob(k, phase))
                                       if self.transition_prob(k, phase)!=0 else -np.inf for k in range(7)])
                max_val = vec_to_max.max()
                #pointer from phase "phase" at day i:
                ptr[phase, i] = vec_to_max.argmax()
                #viterbi update
                logv[phase, i] = log_emit + max_val
        #finish (L+1 must be in phase 7)
        vec_to_max = np.array([logv[k, L] + np.log(self.transition_prob(k, 7))
                               if self.transition_prob(k, 7)!=0 else -np.inf for k in range(7)])
        logprob_data_and_viterbi_path = vec_to_max.max()
        ptr[7, L+1] = vec_to_max.argmax()
        
        #do pointer backtrace
        viterbi_path = np.zeros(L+2)
        viterbi_path[L+1] = 7
        for i in range(L,-1,-1):
            #print(viterbi_path[i+1])
            viterbi_path[i] = ptr[int(viterbi_path[i+1]), i+1]
        #print(logv)
        return({'path':viterbi_path, 'prob':logprob_data_and_viterbi_path})   

    def iterate_updates(self, lhfile, pdgfile, e3gfile, lhfile_withnans, pdgfile_withnans, e3gfile_withnans):
        #use withnans files to make a dict of username+cycleid --> list of T/F Nan or not
        outfile = 'output/stoch_EM_updates.txt'
        out = open(outfile,'w')
        out.close()

        for i in range(100):#change to a convergence check?
            self.parameter_update(lhfile, pdgfile, e3gfile, iter=i, outfilename = outfile) 
        out.close()

    def parameter_update(self, lhfile, pdgfile, e3gfile, iter, outfilename):
        #implement stochastic EM
        #prediction_df is a dataframe with columns for hub_id, cycle_id, day, single stoch draw, hormone vals
        #reset self.df_regression_params and self.Transition for use in future cycles
        outfile = open(outfilename,'a')
        #run current HMM with one stochastic backtrace for each curated baseline cycle
        lh_df = pd.read_csv(lhfile)
        pdg_df = pd.read_csv(pdgfile)
        e3g_df = pd.read_csv(e3gfile)
        L = zip(lh_df['hub_id'].tolist(), lh_df['cycle_index'].tolist())
        
        #add pseudocounts to transitions (i-->i and i-->i+1)
        transition_counts = np.matrix([[0,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,1,1,0,0,0],
                       [0,0,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,1,1]])
        lh_emission_values = [[] for i in range(7)] #list of values (current-prev, i.e. intercept) for each phase 0-7 (0 and 7 will be empty)
        pdg_emission_values = [[] for i in range(7)] 
        e3g_emission_values = [[] for i in range(7)] 
        #CHANGE BACK!!
        #for x in list(L)[:10]:
        for x in L:
            print(x)
            lh_vals = lh_df.loc[(lh_df['hub_id']==x[0]) & (lh_df['cycle_index']==x[1])]
            pdg_vals = pdg_df.loc[(pdg_df['hub_id']==x[0]) & (pdg_df['cycle_index']==x[1])]
            e3g_vals = e3g_df.loc[(e3g_df['hub_id']==x[0]) & (e3g_df['cycle_index']==x[1])]
            cycle_length = lh_vals['cycle_length'].iloc[0]
            #print(x)
            #print(cycle_length)
            lh_vals = lh_vals[['day'+str(i)+'_LH' for i in range(cycle_length)]].iloc[0].to_list()
            pdg_vals = pdg_vals[['day'+str(i)+'_PDG' for i in range(cycle_length)]].iloc[0].to_list()
            e3g_vals = e3g_vals[['day'+str(i)+'_E3G' for i in range(cycle_length)]].iloc[0].to_list()
            lh_diffs = [np.nan]+list(np.array(lh_vals[1:])-np.array(lh_vals[:-1]))
            pdg_diffs = [np.nan]+list(np.array(pdg_vals[1:])-np.array(pdg_vals[:-1]))
            e3g_diffs = [np.nan]+list(np.array(e3g_vals[1:])-np.array(e3g_vals[:-1]))
            logf = self.forward(lh_vals, pdg_vals, e3g_vals)['logf']
            sb = self.single_stoch_backtrace(logf)
            if np.isnan(sb).sum()==0:
            #add to transition counts
                for i in range(len(sb)-1):
                    start = sb[i]
                    end = sb[i+1]
                    transition_counts[int(start),int(end)] += 1
                #strip off phases 0 and 7 from ends
                sb = sb[1:-1]
                temp_df = pd.DataFrame({'stoch_back':sb,'lh_diffs':lh_diffs, 'pdg_diffs':pdg_diffs, 'e3g_diffs':e3g_diffs})
                #print(temp_df)
                #use temp_df to get emission values
                for i in range(1,7):
                    lh_emission_values[i] += temp_df.loc[temp_df['stoch_back']==i]['lh_diffs'].tolist()
                    pdg_emission_values[i] += temp_df.loc[temp_df['stoch_back']==i]['pdg_diffs'].tolist()
                    e3g_emission_values[i] += temp_df.loc[temp_df['stoch_back']==i]['e3g_diffs'].tolist()
       
        #normalize the transition matrix and set (should also compare to old)
        old_T = self.Transition
        self.Transition = transition_counts/transition_counts.sum(axis=1)
        print('transition difference sum')
        print((abs(old_T-self.Transition)).sum()) #check this for convergence
        np.save('output/iterations_transitions/iter'+str(iter), self.Transition)
        outfile.write('transition difference sum\n'+str((abs(old_T-self.Transition)).sum())+'\n')
        
        #print(old_T)
        #print(self.Transition)

        #average and sd for new emissions
        phase = []
        hormone = []
        slope = []
        intercept = []
        stdev = []
        for ph in range(1,7):
            phase.append(ph)
            hormone.append('LH')
            slope.append(1)
            intercept.append(np.nanmean(lh_emission_values[ph]))
            #stdev.append(np.nanstd(lh_emission_values[ph]))

            phase.append(ph)
            hormone.append('PDG')
            slope.append(1)
            intercept.append(np.nanmean(pdg_emission_values[ph]))
            #stdev.append(np.nanstd(pdg_emission_values[ph]))
            
            phase.append(ph)
            hormone.append('E3G')
            slope.append(1)
            intercept.append(np.nanmean(e3g_emission_values[ph]))
            #stdev.append(np.nanstd(e3g_emission_values[ph]))
        old_emissions = self.df_regression_params
        #keep standard deviations the same
        new_df = pd.DataFrame({'Phase':phase, 'Hormone':hormone, 'Slope':slope, 'Intercept':intercept, 'Std Dev':old_emissions['Std Dev']})
        self.df_regression_params = new_df 
        #check intercepts
        merged_df = pd.merge(old_emissions, new_df, on=['Phase','Hormone'])
        merged_df['intercept_diff'] = abs(merged_df['Intercept_x']-merged_df['Intercept_y'])
        print('emission difference sum')
        print(merged_df['intercept_diff'].sum())
        new_df.to_csv('output/iterations_emissions/iter'+str(iter)+'.csv')
        outfile.write('emission difference sum\n'+str((merged_df['intercept_diff'].sum()))+'\n')
        #print(old_emissions)
        #print(self.df_regression_params)

        outfile.close()


if __name__=='__main__':
    #test line
    test = ARHMM('processed/phase_mappings_for_hormones_with_adjusted_slope.csv')
    #first cycle has length 26

    # test_hubid = 'U2F3D9117136337'
    # test_cycle_index = 21
    # dflh = pd.read_csv('processed/lhfw_interp.csv')
    # ix = dflh.loc[(dflh['hub_id']==test_hubid) & (dflh['cycle_index']==test_cycle_index)].index
    # cyc_len = dflh.loc[ix]['cycle_length'].iloc[0]
    # lhvals = dflh.loc[ix][['day'+str(x)+'_LH' for x in range(cyc_len)]].iloc[0].tolist()

    # dfpdg = pd.read_csv('processed/pdgfw_interp.csv')
    # ix = dfpdg.loc[(dfpdg['hub_id']==test_hubid) & (dfpdg['cycle_index']==test_cycle_index)].index
    # cyc_len = dfpdg.loc[ix]['cycle_length'].iloc[0]    
    # pdgvals = dfpdg.loc[ix][['day'+str(x)+'_PDG' for x in range(cyc_len)]].iloc[0].tolist()

    # dfe3g = pd.read_csv('processed/e3gfw_interp.csv')
    # ix = dfe3g.loc[(dfe3g['hub_id']==test_hubid) & (dfe3g['cycle_index']==test_cycle_index)].index
    # cyc_len = dfe3g.loc[ix]['cycle_length'].iloc[0]
    # e3gvals = dfe3g.loc[ix][['day'+str(x)+'_E3G' for x in range(cyc_len)]].iloc[0].tolist()

    # # lhvals = dflh.iloc[(dflh['hub_id']=='U2CF1C916344447') & (df['cycle_index']==32)][['day'+str(x)+'_LH' for x in range(26)]].to_list()
    # # dfpdg = pd.read_csv('processed/pdgfw_interp.csv')
    # # pdgvals = dfpdg.iloc[0][['day'+str(x)+'_PDG' for x in range(26)]].to_list()
    # # dfe3g = pd.read_csv('processed/e3gfw_interp.csv')
    # # e3gvals = dfe3g.iloc[0][['day'+str(x)+'_E3G' for x in range(26)]].to_list()

    # test.multiple_backtraces_and_viterbi(lhvals=lhvals, pdgvals=pdgvals, e3gvals=e3gvals, outfilename='test1',
    #                                      emission_type='E3G')
    F = test.forward(lhvals, pdgvals, e3gvals, emission_type='all')
    print(F['logprob'])
    # B = test.backward(lhvals, pdgvals, e3gvals, emission_type='all')
    # print(B['logprob'])
    # print(test.gamma_probabilities(lhvals, pdgvals, e3gvals, emission_type='all'))
    # print(test.single_stoch_backtrace(F['logf']))
    
    
    test.iterate_updates('processed/lhfw_interp.csv','processed/pdgfw_interp.csv','processed/e3gfw_interp.csv')

    #FIX THIS SO THE UPDATES ONLY USE PRE-IMPUTATION MEASUREMENTS!
    #and maybe include imputation as a step?