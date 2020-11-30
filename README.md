# News Bias Detection

### Overview
Dataset:
All the dataset files are downloadable from here.
https://zenodo.org/record/1489920#.X8R_nBMzZQK

Models:
3 types of models are experimented with. 
  - Classical
  - BiLSTMs
  - BERT

# Code Structure
Each type of model is in its respective folder
  - ./classical_models/tfidf_classical.ipynb
        Contains:
        - Dataloader
        - Linear SVM Classifier
        - SVM with RBF kernel
        - Random Forest Classifier
        - Adaboost Classifer

  - ./bilsm/
        Contains:
        - Bilstm without Attention
            Encoder, Decoder, Attention, Model, dataloader, Train
        - Bilstm with Attention
            Encoder, Decoder, Model, dataloader, Train

  - ./Bert_models/
        Contains:
        - anlp_bert_proj.ipynb
        - anlp_bert_proj_with_title.ipynb

### Reproduce Results

Initallt the dataset files need to be downloaded and put in the prject root directory.
(Link at the top)
  - Classical Models:
    Install dependencies
    ```sh
    pip install scikit-learn
    ```
    Set the root-path as the directory location containing 'byarticle' data files
    Run the Dataloader and tfidf and then simply execute the block corresponding to the required model in the given notebook.
  - BiLSTM with/without Attention:
    Install dependencies
    ```sh
    pip install gensim
    pip install tensorflow
    ```
    Run the dataloader. It saves the file 'processed_data.npz in the same directory'.
    ```sh
    python dataloader.py
    ```
    Then either run the trainer or run from checkpoint.
    ```sh
    python train.py
    python runFromCheckpoint.py
    ```
  - Bert Models
    Install dependencies
    ```sh
    pip install tensorflow
    pip install transformers
    ```
    Sequentially execute the given notebooks for both experiments.
    Please find instructions for loading from checkpoints and processed data in the notebooks themselves.
