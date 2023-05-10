# Fine-tuning RoBERTa Pretrained Model with Adapters for Sentiment Analysis on IMDB Movie Reviews
## Project Description
This project aims to fine-tune the RoBERTa pretrained model from the Hugging Face Transformers library for sentiment analysis on the IMDB movie reviews dataset. The goal of the project is to train a model that can classify each movie review as positive or negative based on the sentiment expressed in the text. The [Adapter](https://arxiv.org/abs/1902.00751) technique is used to add some layers inside the RoBERTa model that are specific to the sentiment analysis task, while keeping the majority of the pretrained weights intact.

## Dataset
The dataset used in this project is the IMDB movie reviews dataset, which contains a collection of 50,000 movie reviews from the IMDB website, labeled as either positive or negative. The dataset is split into 25,000 training examples and 25,000 test examples, with an even distribution of positive and negative reviews in each set.

## Implementation Details
The implementation of the RoBERTa model with Adapters is done using the PyTorch deep learning framework and the Transformers library. The RoBERTa model is fine-tuned on the IMDB movie reviews dataset, using the Adapter technique to add sentiment-specific layers to the model. The adapter modules are added to the base RoBERTa model and trained on the sentiment analysis task while freezing the majority of the pretrained weights.

# Results
| Model             | Acc.        | Precision.        | Recall.        | F1.        |
| ----------------- | ----------- |
| [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)         | 94.16%      | 93.82%      | 94.54%      | 94.18%      |
