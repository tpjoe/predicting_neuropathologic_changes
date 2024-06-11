This folder contains all the files you need to preprocess from raw data received from NACC (example shown in df_samples.csv) to become data suitable for the input/output for the model.

To use it,

1. Make sure you have the dataframe in a format looking like df_samples.csv. This should be the standard data received from NACC.
2. Run the Jupyter Notebook
3. The output (X_UDS_caseII) in the last cell will have the values preprocessed suitable for the model, you may need to subset to certain columns to run in the model

Note the package versions are
1. Python == 3.10.14
2. Panda == '1.5.3'
3. Numpy == '1.26.4'

