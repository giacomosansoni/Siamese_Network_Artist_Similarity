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

![Alt text](/images/siamese_net.png)

## Results
The trained models are capable of distinguishing between similar and dissimilar artists with a high degree of accuracy. Notable observations include:
- Better performance with embeddings combined from image and caption data.
- Model variations and configurations can significantly affect the learning outcomes.

![Alt text](/images/tabella_full_models.png)

## t-SNE Visualization Analysis

The repository includes t-SNE plots that provide a visual understanding of how similar artists cluster together based on their embeddings, which are derived from both image and caption data. These plots are crucial for interpreting the effectiveness of the embedding strategies and the subsequent training of the Siamese networks.

### Key Observations from t-SNE Plots:

- **Dimensionality Reduction**: t-SNE is used to reduce the high-dimensional embedding spaces into 2-dimensional plots that are easier to visualize and interpret.
- **Cluster Formation**: The plots demonstrate clear clustering patterns considering the 'nationality' and the 'genre' of the artists, indicating that the model effectively groups similar artists together. This is essential for the task of artist similarity, where the goal is to identify and group artists with similar styles or genres.
- **Impact of Different Data Types**: The plots illustrate differences in clustering when using embeddings derived from images compared to those from captions. For instance, image-based embeddings tend to cluster by visual style, while caption-based embeddings reflect textual context which might include the nationality.
- **Model Comparisons**: Different models and their configurations can be visually compared in terms of how well they manage to cluster similar artists. This gives insights into which models are more effective for the artist similarity task based on the structure of the data.

These visualizations serve as a direct method to qualitatively assess the performance and behavior of our models, complementing the quantitative metrics also provided in this repository.

![t-SNE Plot Example](/images/tsne.png)

*The example above illustrates a t-SNE plot generated using cosine distance, highlighting clusters based on artist nationality and genre. Such visualizations are instrumental in confirming the intuitive grouping of artists and the effectiveness of the embedding process.*


## Files in the Repository
- **Bert_SBert.ipynb**: I tried different approaches at creating word/sentence embeddings to better understand the ability of different model (Bert, Sbert, Roberta) to create meaningful representations of language. This in order to then choose how to extract the embeddings from the captions of the instagram posts.
- **Embeddings_Posts_Captions.ipynb**: I extract the embeddings of every image (using the Clip model) and every caption (in 4 different ways).
- **min_max_scaling.ipynb**: It contains the code used to normalize our data in order to then create a dataset of homogenous dimension.
- **Triplets.ipynb**: How I created the triplets (anchor, positive, negative) of artists to then train the Siamese Network.
- **Siamese Network.ipynb**: Implementation of the cnn network and the siamese model then trained with a triplet loss.
- **tsne.ipynb**: Creation of 256-dim embeddings representing every artist using the trained models. And visualization in 2 dimensions through tsne of the artists. 
