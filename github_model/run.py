###### import library
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import pickle
import pandas as pd
import argparse
import sys

def experiment(
        exp_args
):
       ###### predefined variables
       categ_features = ['MARISTAT', 'NACCLIVS', 'RESIDENC', 'CBSTROKE', 'HYPERTEN', 'HEARING',
              'HEARAID', 'HEARWAID', 'COGMEM', 'COGVIS', 'NACCCOGF', 'BEVHALL',
              'BEPERCH', 'FRSTCHG', 'DEMENTED', 'NACCPPA', 'NACCPPME', 'NACCBVFT',
              'NACCLBDS', 'NACCALZD', 'NACCALZP', 'NACCLBDE', 'NACCLBDP', 'PSPIF',
              'CORTIF', 'STROKE', 'PRIONIF', 'NACCETPR', 'NACCFTDM', 'NACCIDEM',
              'NACCAPOE', 'NACCNE4S']

       cont_features = ['NACCID', 'NACCVNUM', 'MEMORY', 'ORIENT', 'JUDGMENT', 'CDRSUM',
              'CDRGLOB', 'DECAGE', 'MMSEORDA', 'LOGIMEM', 'MEMUNITS', 'TRAILB',
              'WAIS', 'NACCAGEB', 'NACCAGE', 'NACCUDSD', 'NACCDAGE', 'DONEPEZIL']

       colnames = ['ADNC', 'NACCDIFF', 'NACCNEUR', 'NACCBRAA', 'NACCLEWY','NACCAMY', 'NACCARTE', 'NPHIPSCLSCL', \
              'NACCINF', 'NACCHEM', 'NACCMICR', 'NPWMR', 'NACCAVAS'] + ['NPTDP'+i for i in ['B', 'C', 'D', 'E']]


       ###### load data
       input_dir = exp_args['filename']
       # input_dir = 'example_data.csv'
       data = pd.read_csv(input_dir)

       if pd.Series([i not in data.columns for i in cont_features+categ_features]).any():
              print ('Column names are not correct, please see the list of column names in example_data.csv')
              quit()

       ###### preprocess data
       with open('../prelim_results/Github_model/models/encoder', 'rb') as pickle_file:
              ohe = pickle.load(pickle_file)


       ###### one hot encode data
       data_categ = ohe.transform(data[categ_features]).toarray()
       feature_labels = ohe.categories_
       feature_columns = []
       for ii, feat in enumerate(categ_features):
              feature_columns += [feat+'_'+str(j) for j in feature_labels[ii]]

       data_categ = pd.DataFrame(data_categ, columns=feature_columns)
       # data_categ.index = X_UDS_caseII.index
       data_transformed = pd.concat([data[cont_features], data_categ], axis=1)


       ###### get number of features
       n_features = data_transformed.shape[1] - 2 


       ###### reshape matrix
       data_transformed = data_transformed.sort_values(['NACCID', 'NACCVNUM'])
       data_transformed_pad = data_transformed.copy()
       data_transformed_pad = data_transformed_pad.set_index(['NACCID', 'NACCVNUM'])

       n_patients = data_transformed_pad.groupby('NACCID').count().shape[0]
       x_lens = torch.Tensor(data_transformed_pad.groupby('NACCID').count().iloc[:,0].tolist()).reshape([n_patients, -1])
       data_transformed_pad = data_transformed_pad.reindex(pd.MultiIndex.from_product([data_transformed_pad.index.levels[0], \
                     data_transformed_pad.index.levels[1]], names=['NACCID', 'NACCVNUM']), fill_value=0) # padding 'PAD' with 0

       max_visit = data_transformed_pad.index.levels[1].max()
       code_mat = (data_transformed_pad.to_numpy().reshape([int(data_transformed_pad.shape[0]/max_visit), max_visit*(n_features)]))
       code_mat = torch.Tensor(code_mat.reshape([code_mat.shape[0], -1, code_mat.shape[1]]))
       code_mat = code_mat.reshape(int(code_mat.shape[0]), max_visit, int(code_mat.shape[2]/max_visit))
       n_obs = len(x_lens)
       x_holdout = pack_padded_sequence(code_mat, x_lens.view(n_obs, -1).numpy().flatten(), batch_first=True, enforce_sorted=False)


       ###### load model
       model = torch.load('models/heroku_withAge_seed1_Clin_heroku_lstm_variable')
       model.eval()


       ###### make prediction and output
       holdout = model(x_holdout, 0, 0)
       holdout_ = [pd.Series(i.cpu().detach().numpy().flatten()) for i in holdout]
       predicted_values = pd.concat(holdout_, axis=1)
       predicted_values.columns = colnames
       predicted_values.index = data.NACCID.sort_values().unique()
       predicted_values.to_csv('predicted_output.csv')


def run_experiment(
        experiment_params,
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str,
                        help='Input csv file directory')
    exp_args = parser.parse_args(sys.argv[1:])
    
    
    experiment_params['exp_args'] = vars(exp_args)
    experiment( **experiment_params)

if __name__ == '__main__':
       experiment_params = dict()
       run_experiment(experiment_params)