# Medical-Domain-Specific-Language-Models

This repositary contains code used for following the PhD thesis and publications:

1. Yogarajan, V (2021). Domain-specific Language Models for Multi-label Classification of Medical Text. The University of Waikato. PhD Thesis. (examination process)
2. Yogarajan, V., Montiel J., Smith T., & Pfahringer B. (2021) Transformers for Multi-label Classification of Medical Text: An Empirical Comparison. In: Tucker A., Henriques Abreu P., Cardoso J., Pereira Rodrigues P., Riaño D. (eds) Artificial Intelligence in Medicine. AIME 2021. Lecture Notes in Computer Science, vol 12721. Springer, Cham. [link](https://doi.org/10.1007/978-3-030-77211-6_12)
3. Yogarajan, V., Gouk, H., Smith, T., Mayo, M., & Pfahringer, B. (2020). Comparing High Dimensional Word Embeddings Trained on Medical Text to Bag-of-Words for Predicting Medical Codes. In Asian Conference on Intelligent Information and Database Systems. Springer, Cham, pp. 97-108. [pdf](https://drive.google.com/file/d/1hnjaGK-egbN0INlxD8GW0olgQAYQUgAX/view) 
4. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2020). Seeing The Whole Patient: Using Multi-Label Medical Text Classification Techniques to Enhance Predictions of Medical Codes. [arXiv preprint arXiv:2004.00430](https://arxiv.org/abs/2004.00430). 
5. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Concatenated BioMed-Transformers for Multi-label Classification of Medical Text. (under submission)
6. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Predicting COVID-19 Patient Shielding: A Comprehensive Study. (under submission).

# Multi-label Problems

|Classification Problem | Data  | L  | Inst   |  Data  | L  | Inst  |  
| :------ | --------: | --------: | -----: | --------: | --------: | -----: |
|ICD-9 Level 1 | MIMIC-III |   18 | 52,722   |  eICU | 18 |154,808   | 
|ICD-9 Level 2 | MIMIC-III | 158 | 52,722   |  eICU  | 93 |154,808     | 
|ICD-9 Level 3 | MIMIC-III | 923 | 52,722   |  eICU  | 316 | 154,808   | 
|Cardiovascular | MIMIC-III | 30 | 28,154   |  eICU | 15 | 53,477 | 
|COVID-19 | MIMIC-III | 42|35,458   |  eICU  | 25 | 34,387 |
|Fungal or bacterial | MIMIC-III | 73 | 30,814   |  eICU | 42|54,193|     

# Examples
1. [Binary classification using GRU and pre-trained embeddings](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Binary_classification/Binary_classification_GRU.ipynb)
2. [Binary classification using PubMedBERT](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Binary_classification/Binary_classification_PubMedBERT.ipynb)
3. [Vizualising simillar words using pre-trained embeddings](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/fastText_Embeddings/vizualise_wiki.ipynb)
4. [Multi-label classification using CNNText and pre-trained embeddings](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Multilabel/icd9_cnntext.ipynb)
5. [Multi-label classification using HAN(GRU) or HAN(LSTM) and pre-trained embeddings](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Multilabel/HAN.py)
6. [Multi-label classification using Longformer](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Multilabel/longformer_fungal_eICU.py)
7. [Multi-label classification using BioMed-RoBERTa](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Multilabel/Biomed-Roberto_cardiology.py)
8. [CD-plot and Nemenyi test using Python](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Multilabel/posthoc-nemenyi.ipynb)

# Open-source Frameworks 

Transformer implementations are based on the open-source PyTorch-transformer repositories [Huggingface](https://github.com/huggingface/transformers) and [Simple Transformers](https://simpletransformers.ai/). 

Transformer models used include:[BERT-base](https://github.com/google-research/bert),[Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT),[BioMed-RoBERTa](https://huggingface.co/allenai/biomed_roberta_base),[PubMedBERT](https://microsoft.github.io/BLURB/models.html),[MeDAL-Electra](https://github.com/BruceWen120/medal),[Longformer](https://github.com/allenai/longformer) and [TransformerXL](https://github.com/kimiyoung/transformer-xl)

Neural network models presented are implemented using [PyTorch](https://github.com/pytorch/pytorch) and [Keras/Tensorflow](https://www.tensorflow.org). 

Traditional classifiers such as logistic regression, random forest, and classifier chains use implementations of [the Waikato Environment for Knowledge Analysis (WEKA)](https://www.cs.waikato.ac.nz/ml/weka/) framework for binary classification and [MEKA](http://waikato.github.io/meka/) for multi-label classification.
 
Evaluations were done using [sklearn metrics](https://scikit-learn.org/stable/modules/classes.html\#module-sklearn.metrics). 
