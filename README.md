# Siamese Network For Artist Similarity

## Files in the Repository
- **Bert_SBert**: I tried different approaches at creating word/sentence embeddings to better understand the ability of different model (Bert, Sbert, Roberta) to create meaningful representations of language. This in order to then choose how to extract the embeddings from the captions of the instagram posts.
- **Embeddings_Posts_Captions**: I extract the embeddings of every image (using the Clip model) and every caption (in 4 different ways).
- **min_max_scaling**: It contains the code used to normalize our data in order to then create a dataset of homogenous dimension.
- **Triplets**: How I created the triplets (anchor, positive, negative) of artists to then train the Siamese Network.
- **Siamese Network**: Implementation of the cnn network and the siamese model then trained with a triplet loss. 
