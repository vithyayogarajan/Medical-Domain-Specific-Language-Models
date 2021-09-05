# Multi-Label Classification

<img src="https://user-images.githubusercontent.com/60803118/132111996-dd029e2c-5d1e-4cb6-b2ef-70180f9d479d.jpeg" alt="injury-nurse-in-hospital-cartoon" width="200"/>


**Single-label classification**: Is this a picture of a hospital bed? 

**Multi-label classification**: Which of the labels are relevant to the picture? 
{broken leg, injury, patient, nurse, hospital, unconscious}

i.e., each instance can have **multiple** labels instead of **single one**.
    
## Classifiers and Language Models

**'Traditional' classifiers** such as Binary Relevance (BR), Classifier Chains (CC) and Multi-label k-nearest neighbor classifier (MLkNN) use [MEKA](http://waikato.github.io/meka/). Example scripts are provided above. 

 **Pre-trained word embeddings based Neural Networks** 
 1. [CAML and DRCAML](https://github.com/jamesmullenbach/caml-mimic)  
 2. [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) example code provided. 
 3. [CNNText](https://arxiv.org/abs/1408.5882) example code provided. 
 
## Transformers
Models used include:[BERT-base](https://github.com/google-research/bert),[Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT),[BioMed-RoBERTa](https://huggingface.co/allenai/biomed_roberta_base),[PubMedBERT](https://microsoft.github.io/BLURB/models.html),[MeDAL-Electra](https://github.com/BruceWen120/medal),[Longformer](https://github.com/allenai/longformer) and [TransformerXL](https://github.com/kimiyoung/transformer-xl)
