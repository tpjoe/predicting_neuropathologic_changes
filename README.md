# predicting_neuropathologic_changes
This repository contains a trained multitask LSTM model for predicting 17 neuropathologic changes from clinical features structured as shown in the template. The input is structured as in NACC Dataset (https://naccdata.org/). It requires 50 variables from the dataset. These 50 variables, as well as the input structure, can be found in the "example_data.csv".

Breifly, the input of the structure is as follow:

Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

To use the model you would need pyTorch version 1.9.0

To run the model, simply type the command with an argument "-f" pointing to the input file, for example

```
python run.py -example_data.csv
```

This will output the prediction table "predicted_output.csv", containing predicted values for all 17 neuropathologic changes. Note that the predicted values is the model's relative probability ranging from 0 to 1.
