# predicting_neuropathologic_changes
This repository contains a trained multitask LSTM model for predicting 17 neuropathologic changes from clinical features structured as shown in the template. The input is structured as in NACC Dataset (https://naccdata.org/). It requires 50 variables from the dataset. These 50 variables, as well as the input structure, can be found in the "example_data.csv".

Breifly, the input table must have the following structure:

NACCID | NACCVNUM | Feature_1 | Feature_2 | Feature_... | Feature_50 | 
--- | --- | --- | --- |--- |---|
ID1 | 1 | value | value | value | value |
ID1 | 2 | value | value | value | value |
ID1 | 3 | value | value | value | value |
ID2 | 1 | value | value | value | value |
ID2 | 2 | value | value | value | value |
ID3 | 1 | value | value | value | value |
ID3 | 2 | value | value | value | value |
ID3 | 3 | value | value | value | value |
ID3 | 4 | value | value | value | value |
--- | --- | --- | --- |--- |---|

To use the model you would need pyTorch version 1.9.0

To run the model, simply type the command with an argument "-f" pointing to the input file, for example

```
python run.py -example_data.csv
```

This will output the prediction table "predicted_output.csv", containing predicted values for all 17 neuropathologic changes. Note that the predicted values is the model's relative probability ranging from 0 to 1.

Update [05/12/23] - When using the model, DECAGE should be replaced with 0 when unknown for 888.
