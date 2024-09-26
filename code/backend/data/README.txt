The dataset will be contained in this folder and spread to the corresponding sub-folders :

- raw is the entire dataset before pre-processing, structured like this : raw/human, raw/ai and raw/unknown, contain .txt files
- processed : dataset after filtering and optimization (tokenization, case-insensitive, punctuation and special characters, stopwords, metadata)
- test : dataset used for final evaluation after training and corresponding validation process
- training : dataset used for training process
- validation : dataset used during the development of models to validate the results obtained after training