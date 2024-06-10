# EEC273_final-project

We use 3 different models including LSTM, GNN, Transformer to predict the network traffic. And we use 80% of data to train our model and 20% of data to test our model. Then, we use loss Mean Square Error(MSE) and test Mean Square Error(MSE) to compare these 3 different models' prediction accuracy.

The dataset 'sms-call-internet-mi-2014-01-01' is from 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:%2010.7910/DVN/EGZHFV' website.

sms-call-internet-mi-2014-01-01_1000 : we fetch the first 1000 rows of 'sms-call-internet-mi-2014-01-01' data since the whole file size is too big(over 25MB) to upload on GitHub.

sms-call-internet-mi-2014-01-01_10000 : we fetch the first 10000 rows of 'sms-call-internet-mi-2014-01-01' data.

LSTM.py : we use LSTM model to predict the network traffic

GNN.py : we use GNN model to predict the network traffic

Transformer.py : we use Transformer model to predict the network traffic
