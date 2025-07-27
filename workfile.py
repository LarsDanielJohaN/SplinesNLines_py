
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
import pandas as pd
from astropy.table import Table 
from astropy.io import fits
import matplotlib.pyplot as plt 
import torch
import numpy as np 
import SplinesNLines 



np.random.seed(201763)

all_data = pd.read_csv('Selection2_les_astro_guapes.csv') #Reads record data. 
all_data = all_data.drop_duplicates(subset= ['sdss_id'])
file_names = all_data['sas_file'].to_list() #Places all file names in a list. 
f1 = file_names[0]

"""
Note: Line profile information is read first to see which line profile information is available and how it varies. 
"""
#Read information for line profiles-----------------------------------------------------------------------------------------------------------
cols = ['LINENAME', 'LINEWAVE', 'LINEZ', 'LINEZ_ERR', 'LINESIGMA', 'LINESIGMA_ERR' ]

f1 = file_names[0]
table_line = Table.read(f1, hdu = 4)
table_line = table_line.to_pandas()
table_line = table_line[cols]
table_line['file_name'] = f1
no_prof = [] #Array to store the number of profiles in each file. 

var = [] #Adds to array the names of the profiles of those files with 29 recorded profiles. 
max_lam =[] #Array to store the maximum recorded value of wavelength. 
min_lam = []  #Array to store the minimum recorded value of wavelength. 
var_pos = []


"""
Index(['PLATE', 'MJD', 'FIBERID', 'LINENAME', 'LINEWAVE', 'LINEZ', 'LINEZ_ERR',
       'LINESIGMA', 'LINESIGMA_ERR', 'LINEAREA', 'LINEAREA_ERR', 'LINEEW',
       'LINEEW_ERR', 'LINECONTLEVEL', 'LINECONTLEVEL_ERR', 'LINENPIXLEFT',
       'LINENPIXRIGHT', 'LINEDOF', 'LINECHI2'],
      dtype='object')
Index(['PLATE', 'MJD', 'FIBERID', 'LINENAME', 'LINEWAVE', 'LINEZ', 'LINEZ_ERR',
       'LINESIGMA', 'LINESIGMA_ERR', 'LINEAREA', 'LINEAREA_ERR', 'LINEEW',
       'LINEEW_ERR', 'LINECONTLEVEL', 'LINECONTLEVEL_ERR', 'LINENPIXLEFT',
       'LINENPIXRIGHT', 'LINEDOF', 'LINECHI2'],
      dtype='object')

"""

for i in range(1, len(file_names)):

    oth_table = Table.read(file_names[i], hdu = 1).to_pandas()
    try:
        curr_table = Table.read(file_names[i], hdu = 4).to_pandas()
        curr_table = curr_table[cols]
        #log_lam_curr = 10**oth_table['LOGLAM']
        log_lam_curr = oth_table['LOGLAM']

    except:
        curr_table = Table.read(file_names[i], hdu = 3).to_pandas()
        curr_table = curr_table[cols]
        log_lam_curr = 10**oth_table['loglam']
        log_lam_curr = oth_table['loglam']


    max_lam.append(log_lam_curr.max())
    min_lam.append(log_lam_curr.min())
         
    curr_table['file_name'] = len(curr_table)*[file_names[i]]
    if len(curr_table) == 29:
        var.append(curr_table['LINENAME'].to_list())
        var_pos.append(curr_table['LINEWAVE'].to_list())
    no_prof.append(len(curr_table))
    table_line = pd.concat([table_line, curr_table])

print("Nan values per column:   \n",   table_line.isnull().sum())
line_sd_mean = table_line.groupby('LINENAME')['LINESIGMA'].mean()

table_line = table_line.sort_values(['file_name','LINENAME'])
to_use_profiles = var[0]
print(table_line)

to_use_profiles = list(dict.fromkeys([(x.decode('utf-8').strip()).replace(' ', '') 
                                      if isinstance(x, bytes) else (x.strip()).replace(' ', '') for x in to_use_profiles
                                    ])) #Convert to_use_profiles elements to strings and remove spaces. 
table_line['LINENAME'] = table_line['LINENAME'].str.decode('utf-8') #Change the format from bytes to string. 
table_line['LINENAME'] = table_line['LINENAME'].str.replace(' ', '') #Replace all of spaces with empty strings. 


table_line = table_line[ (table_line['LINESIGMA'] > 0) & (table_line['LINENAME'].isin(to_use_profiles)) ] #Select only those with LINESIGMA> 0 and that 
                                                                                                          #correspond to LINENAME's that are going to be used. 
plt.hist(no_prof)
plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------------------






unique_pairs = (table_line[['LINENAME', 'LINEWAVE']].drop_duplicates().groupby('LINENAME', as_index=False).agg(LINEWAVE=('LINEWAVE', 'mean')))
line_sig_mean = (table_line.groupby('LINENAME', as_index= False).agg(LINESIGMA = ('LINESIGMA', 'mean'))  )
line_df = unique_pairs.merge(line_sig_mean, on = 'LINENAME')

line_wave = np.sort(line_df['LINEWAVE'].to_numpy())


all_line_data = {
    'LINEWAVE': [1215.67, 1240.81, 1549.48, 1640.42, 1908.734, 2800.3152, 3727.0917, 
                 3729.8754, 3869.8568, 3890.1511, 3971.1232, 4102.8916, 4341.6843, 
                 4364.4353, 4686.9915, 4862.6830, 4960.2949, 5008.2397, 5413.0245, 
                 5578.8878, 5756.1862, 5877.3086, 6302.0464, 6363.776, 6313.8056, 
                 6365.5355, 6549.8590, 6564.6140, 6585.2685, 6718.2943, 6732.6782, 
                 7137.7572],
    'LINENAME': ['Ly_alpha', 'N_V 1240', 'C_IV 1549', 'He_II 1640', 'C_III] 1908', 
                 'Mg_II 2799', '[O_II] 3725', '[O_II] 3727', '[Ne_III] 3868', 
                 'H_epsilon', '[Ne_III] 3970', 'H_delta', 'H_gamma', '[O_III] 4363', 
                 'He_II 4685', 'H_beta', '[O_III] 4959', '[O_III] 5007', 'He_II 5411', 
                 '[O_I] 5577', '[N_II] 5755', 'He_I 5876', '[O_I] 6300', '[O_I] 6363', 
                 '[S_III] 6312', '[O_I] 6363', '[N_II] 6548', 'H_alpha', '[N_II] 6583', 
                 '[S_II] 6716', '[S_II] 6730', '[Ar_III] 7135'],
    'ZINDEX': ['z_lya', 'zemission', 'zemission', 'zemission', 'zemission', 'zemission', 
               'zemission', 'zemission', 'zemission', 'zemission', 'zemission', 
               'zemission', 'zemission', 'zemission', 'zemission', 'zemission', 
               'zemission', 'zemission', 'zemission', 'zemission', 'zemission', 
               'zemission', 'zemission', 'zemission', 'zemission', 'zemission', 
               'zemission', 'zemission', 'zemission', 'zemission', 'zemission', 
               'zemission'],
    'WINDEX': ['w_ly_a', 'w_n_v', 'wemission', 'wemission', 'wemission', 'wemission', 
               'wemission', 'wemission', 'wemission', 'w_balmer', 'wemission', 
               'w_balmer', 'wemission', 'wemission', 'wemission', 'w_balmer', 
               'wemission', 'wemission', 'wemission', 'wemission', 'wemission', 
               'wemission', 'wemission', 'wemission', 'wemission', 'wemission', 
               'wemission', 'w_balmer', 'wemission', 'wemission', 'wemission', 
               'wemission']
}

all_line_data = pd.DataFrame(all_line_data)
line_data_use = all_line_data[(all_line_data['LINEWAVE'] > np.min(min_lam)) & (all_line_data['LINEWAVE'] < np.max(max_lam))]



B_f = 80 #Sets number of B-Spline functions to use. 
m = 2 #Value such that m+1 is the degree of the basis B-Spline functions. Needs: B_f + m + 2 knots. 
T = np.sort( np.concatenate([line_data_use['LINEWAVE'].to_numpy(), np.linspace(np.min(min_lam), np.max(max_lam), B_f + m + 2 - len(line_data_use['LINEWAVE'].to_numpy())   )  ]    ))


to_use_profiles_set = set(to_use_profiles)
X_a3 = []
f_a3 = []
T_a3 = []
To_a3 = []
not_used = []

for o in range(len(file_names)):
    if o%100 == 0:
        print(f"Going for file {o}")

    f = file_names[o]
    curr_file = Table.read(f, hdu = 1).to_pandas() #Reads the .fits file corresponding to the Flux-wavelength data for f. 

    #For the current file, select those points without specified problems and with a stricty possitive precision parameter (IVAR).
    #The try-except helps to handle both .fits files from DR17 and DR19 without breaking the flow of execution. 
    try:
        curr_file = curr_file[(curr_file['AND_MASK'] == 0) & (curr_file['IVAR'] > 0)]
        tau_ok_name = 'IVAR'
        log_lam_name = 'LOGLAM'
        flux_name = 'FLUX'
        #curr_file_line = Table.read(f, hdu = 4).to_pandas()

    except:
        curr_file = curr_file[(curr_file['and_mask'] == 0) & (curr_file['ivar'] > 0)]
        tau_ok_name = 'ivar'
        log_lam_name = 'loglam'
        flux_name = 'flux'
        #curr_file_line = Table.open(f, hdu = 3).to_pandas()


    try:
        select_idx = [( (i+1)%np.ceil( len(curr_file)/1000) == 0  ) for i in range(len(curr_file)) ] #Reduces the number of points to use. 
        curr_file = curr_file[select_idx] #Selects the points to use. 
        #line_prof_curr = table_line[table_line['file_name'] == f]#Selects the line profile information for f. 
        #line_prof_curr = line_prof_curr[['LINENAME', 'LINEWAVE', 'LINESIGMA']] 

        #missing_line_prof =  list(to_use_profiles_set - set(line_prof_curr['LINENAME'].to_list())) #Checks which line profiles are missing. 
        #if len(missing_line_prof) > 0:
        #    line_prof_curr = pd.concat([line_prof_curr, line_df[line_df['LINENAME'].isin(missing_line_prof)]])
        #line_prof_curr = line_prof_curr.sort_values('LINEWAVE')        
        Tau_ok = curr_file[tau_ok_name].to_numpy()
        To = 10**(curr_file[log_lam_name].to_numpy())
        X_B = SplinesNLines.get_basis_mat_B_Spline_opt(To, B_f, m, T)
        #X_L = SplinesNLines.eval_Line_Profiles_opt(To, line_wave, line_prof_curr['LINESIGMA'].to_numpy() ) #Falta width values
        fo = (curr_file[flux_name].to_numpy()).reshape(-1,1)
        To_a3.append(To)
        X_a3.append(X_B)
        #X_a.append(np.hstack((X_B, X_L)))
        T_a3.append(np.diag(curr_file[tau_ok_name].to_numpy()))
        f_a3.append(fo)
    except:
        print(f"Error on file {f}. \n Moving on...")
        not_used.append(f)


A = SplinesNLines.get_mat_comp(X_a3, f_a3, T_a3)
del X_a3 #, f_a3, T_a3 
L = SplinesNLines.EM_alg(A['XtXf'], A['XtTX'], n_max = 10000, tol = 1e-10)
mu_h = L['mu_h']
S_h = L['S_h']

T_mu = np.linspace(np.mean(min_lam), np.mean(max_lam), num = 1000)
X_mu = SplinesNLines.get_basis_mat_B_Spline_opt(T_mu, B_f, m, T)
mu = torch.from_numpy(X_mu)@mu_h



line_waves = line_data_use['LINEWAVE'].to_numpy()
line_names = line_data_use['LINENAME'].to_list()
mu = torch.from_numpy(X_mu)@mu_h
plt.plot(T_mu, mu)

for i in range(len(line_waves)):
    lw = line_waves[i]
    idx_T = np.searchsorted(T_mu, lw)
    plt.axvline(x = lw, color = 'red', alpha = 0.3)
    #plt.text(x = lw, y = torch.mean(  mu[idx_T - 3:idx_T + 3]   ), s = line_names[i])

plt.show()


l = np.linspace(np.min(min_lam), np.max(max_lam), num = 2000 )
X_g = torch.from_numpy(SplinesNLines.get_basis_mat_B_Spline_opt(l, B_f, m, T))

nr = 4
nc = 3
No = nr*nc
five_galaxies_m0 = L['m0'][0:No, :]
five_f = f_a3[0:No]
five_To = To_a3[0:No]

fig, ax = plt.subplots(nrows = nr,  ncols = nc)
ax = ax.flatten()
colors = plt.cm.tab10.colors[:No]  # First 'n' colors from 'tab10' colormap

for i in range(No):
    mu = X_g@five_galaxies_m0[i, :].T
    ax[i].scatter(five_To[i], five_f[i], alpha = 0.1 , color = 'grey') # color = colors[i])
    ax[i].plot(l, mu, color = 'black')
    ax[i].set_ylim(torch.mean(mu) - 2*torch.std(mu)  ,   torch.mean(mu) + 2*torch.std(mu)   )


plt.show()


mu = X_g@five_galaxies_m0[0, :].T
plt.scatter(five_To[0], five_f[0], alpha = 0.1, color = 'grey')
plt.plot(l, mu, color = 'black')
plt.ylim(torch.mean(mu) - 2*torch.std(mu)  ,   torch.mean(mu) + 3*torch.std(mu))
plt.show()






