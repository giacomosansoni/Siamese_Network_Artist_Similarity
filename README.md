# Artist Similarity Evaluation Using Siamese Networks

## Overview
This repository contains the implementation and evaluation of Siamese networks used to determine artist similarity based on multimodal data (images and textual captions). The project integrates various machine learning techniques, including embedding extraction using CLIP and SBERT, and triplet loss optimization for similarity assessments.

## Features
- **Data Processing**: Scripts to download artist posts, and extract and preprocess embeddings.
- **Model Training**: Training Siamese networks to evaluate artist similarity with options for multiple training configurations.
- **Evaluation**: Code to perform cross-validation and compute metrics such as Average Validity and Recall.

## Models
- **1D-CNN** for time series/sound classification.
- **CRNN**: Combines CNN for feature extraction and RNN for temporal summarization.
- **Siamese Network**: Utilizes triplet loss to learn from artist similarity.

## Files in the Repository
- **Bert_SBert.ipynb**: I tried different approaches at creating word/sentence embeddings to better understand the ability of different model (Bert, Sbert, Roberta) to create meaningful representations of language. This in order to then choose how to extract the embeddings from the captions of the instagram posts.
- **Embeddings_Posts_Captions.ipynb**: I extract the embeddings of every image (using the Clip model) and every caption (in 4 different ways).
- **min_max_scaling.ipynb**: It contains the code used to normalize our data in order to then create a dataset of homogenous dimension.
- **Triplets.ipynb**: How I created the triplets (anchor, positive, negative) of artists to then train the Siamese Network.
- **Siamese Network.ipynb**: Implementation of the cnn network and the siamese model then trained with a triplet loss. 
