# Medical-Domain-Specific-Language-Models

This repositary contains code used for following the PhD thesis and publications:

1. Yogarajan, V (2021). Domain-specific Language Models for Multi-label Classification of Medical Text. The University of Waikato. PhD Thesis. (examination process)
2. Yogarajan, V., Montiel J., Smith T., & Pfahringer B. (2021) Transformers for Multi-label Classification of Medical Text: An Empirical Comparison. In: Tucker A., Henriques Abreu P., Cardoso J., Pereira Rodrigues P., Riaño D. (eds) Artificial Intelligence in Medicine. AIME 2021. Lecture Notes in Computer Science, vol 12721. Springer, Cham. [link](https://doi.org/10.1007/978-3-030-77211-6_12)
3. Yogarajan, V., Gouk, H., Smith, T., Mayo, M., & Pfahringer, B. (2020). Comparing High Dimensional Word Embeddings Trained on Medical Text to Bag-of-Words for Predicting Medical Codes. In Asian Conference on Intelligent Information and Database Systems. Springer, Cham, pp. 97-108.
4. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2020). Seeing The Whole Patient: Using Multi-Label Medical Text Classification Techniques to Enhance Predictions of Medical Codes. [arXiv preprint arXiv:2004.00430](https://arxiv.org/abs/2004.00430). 
5.Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Concatenated BioMed-Transformers for Multi-label Classification of Medical Text. (under submission)
6. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Predicting COVID-19 Patient Shielding: A Comprehensive Study. (under submission).

# Tutorials and Demos 
1. [Binary classification using GRU and pre-trained embeddings](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Binary_classification/Binary_classification_GRU.ipynb)
2. 

# Open-source frameworks 
Models used include:[BERT-base](https://github.com/google-research/bert),[Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT),[BioMed-RoBERTa](https://huggingface.co/allenai/biomed_roberta_base),[PubMedBERT](https://microsoft.github.io/BLURB/models.html),[MeDAL-Electra](https://github.com/BruceWen120/medal),[Longformer](https://github.com/allenai/longformer) and [TransformerXL](https://github.com/kimiyoung/transformer-xl)

Transformer implementations are based on the open-source PyTorch-transformer repositories [Huggingface](https://github.com/huggingface/transformers) and [Simple Transformers](https://simpletransformers.ai/). 

Neural network models presented are implemented using [PyTorch](https://github.com/pytorch/pytorch) and [Keras/Tensorflow](https://www.tensorflow.org). 

Evaluations were done using [sklearn metrics](https://scikit-learn.org/stable/modules/classes.html\#module-sklearn.metrics). 

Traditional classifiers such as logistic regression, random forest, and classifier chains use implementations of [the Waikato Environment for Knowledge Analysis (WEKA)](https://www.cs.waikato.ac.nz/ml/weka/) framework for binary classification and [MEKA](http://waikato.github.io/meka/) for multi-label classification.
 
