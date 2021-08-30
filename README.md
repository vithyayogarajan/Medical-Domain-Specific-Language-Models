# Medical-Domain-Specific-Language-Models

This repositary contains code used for following the PhD thesis and publications:

1. Yogarajan, V (2021). Domain-specific Language Models for Multi-label Classification of Medical Text. The University of Waikato. PhD Thesis. (examination process)
2. Yogarajan, V., Montiel J., Smith T., & Pfahringer B. (2021) Transformers for Multi-label Classification of Medical Text: An Empirical Comparison. In: Tucker A., Henriques Abreu P., Cardoso J., Pereira Rodrigues P., Riaño D. (eds) Artificial Intelligence in Medicine. AIME 2021. Lecture Notes in Computer Science, vol 12721. Springer, Cham. [link](https://doi.org/10.1007/978-3-030-77211-6_12)
3. Yogarajan, V., Gouk, H., Smith, T., Mayo, M., & Pfahringer, B. (2020). Comparing High Dimensional Word Embeddings Trained on Medical Text to Bag-of-Words for Predicting Medical Codes. In Asian Conference on Intelligent Information and Database Systems. Springer, Cham, pp. 97-108.
4. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2020). Seeing The Whole Patient: Using Multi-Label Medical Text Classification Techniques to Enhance Predictions of Medical Codes. [arXiv preprint arXiv:2004.00430](https://arxiv.org/abs/2004.00430). 
5.Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Concatenated BioMed-Transformers for Multi-label Classification of Medical Text. (under submission)
6. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Predicting COVID-19 Patient Shielding: A Comprehensive Study. (under submission).

# Tutorials
1. [Binary classification using GRU and pre-trained embeddings](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Binary_classification/Binary_classification_GRU.ipynb)

## Data and Labels 

Medical Information Mart for Intensive Care (MIMIC) is one of the most extensive publicly available medical databases. It contains de-identified health records of 49,785 adult patient admissions and 7,870 neonates admissions in critical care units at the Beth Israel Deaconess Medical Center between 2001 and 2012. [MIMIC-III Clinical Database](https://physionet.org/works/MIMICIIIClinicalDatabase/access.shtml)

The electronic Intensive Care Unit (eICU) is a database formed from the Philips eICU program, which is a telehealth program delivering information to caregivers and, as a result, enhancing patient care. It contains de-identified data for more than 200,000 patient admitted to ICUs in 2014 and 2015 and monitored by programs across the United States. [eICU Database](https://eicu-crd.mit.edu/)

|Classification Problem | Data  | L  | Inst   |  Data  | L  | Inst  |  
| :------ | --------: | --------: | -----: | --------: | --------: | -----: |
|ICD-9 Level 1 | MIMIC-III |   18 | 52,722   |  eICU | 18 |154,808   | 
|ICD-9 Level 2 | MIMIC-III | 158 | 52,722   |  eICU  | 93 |154,808     | 
|ICD-9 Level 3 | MIMIC-III | 923 | 52,722   |  eICU  | 316 | 154,808   | 
|Cardiovascular | MIMIC-III | 30 | 28,154   |  eICU | 15 | 53,477 | 
|COVID-19 | MIMIC-III | 42|35,458   |  eICU  | 25 | 34,387 |
|Fungal or bacterial | MIMIC-III | 73 | 30,814   |  eICU | 42|54,193|     

Sample codes are provided for MIMIC-III data.


## Health-Related Document Level Embeddings


### Training Embeddings 
Embeddings presented in the research are pre-trained:
- using [fastText](https://fasttext.cc/) 
- at document level
- for both [CBOW](http://dblp.org/rec/bib/journals/corr/abs-1301-3781) (T300 and T600) and [Skip-gram](http://dblp.org/rec/bib/journals/corr/abs-1301-3781) (T300SG, T600SG) models


#### Training Data  

The TREC precision medicine/clinical decision support track 2017 [TREC2017](https://trec.nist.gov/pubs/trec26/papers/Overview-PM.pdf) data (24G of health-related data) is used for pre-training embeddings. This includes 26.8 million published abstracts of medical literature listed on PubMed Central, 241,006 clinical trials documents, and 70,025 abstracts from recent proceedings focused on cancer therapy from AACR (American Association for Cancer Research) and ASCO (American Society of Clinical Oncology).

The clinical notes from [MIMIC-III Clinical Database](https://physionet.org/works/MIMICIIIClinicalDatabase/access.shtml) used for training and experiments. See publications for details of the results. 


| Model | Dimensions | Data | Training Time | Model Size  |
| :------ | --------: |--------: | --------: | -----: |
|M300 | 300 | MIMIC | 1 hour | 5GB |
|T300 | 300 | TREC | 7 hours | 13GB |
|T300SG | 300 | TREC | 28 hours | 13G |
|TM300 | 300 | TREC+MIMIC | 9 hours | 15GB |
|T600 | 600 | TREC | 13 hours | 23GB |
|T600SG | 600 | TREC | 51 hours | 23G |
|TM600 | 600 | TREC+MIMIC | 16 hours | 30GB |
|T900 | 900 | TREC | 19 hours | 35GB |
|TM900 | 900 | TREC+MIMIC | 23 hours | 54GB |


Training time is calculated based on processing run on a 4 core Intel i7-6700K CPU @ 4.00GHz with 64GB of RAM. Parameter choices for training: character n-grams of length 5, a window of size 5, ten negative samples per positive sample, and learning rate of 0.05 is used.

#### Obtaining embeddings from a pre-trained model

1. Word level
 
./fasttext print-word-vectors model.bin < input_text.txt > out_text.txt  

2. Document level
 
./fasttext print-sentence-vectors model.bin < input_text.txt > out_text.txt

#### Neural Networks
Neural network models presented are implemented using [PyTorch](https://github.com/pytorch/pytorch) and [Keras/Tensorflow](https://www.tensorflow.org). 

Evaluations were done using [sklearn metrics](https://scikit-learn.org/stable/modules/classes.html\#module-sklearn.metrics). 



### Vizualising Embeddings

Sample code for visualising word embeddings trained by fastText using general text ([published model](https://fasttext.cc/docs/en/english-vectors.html)) is presented. Most similar words from an example input sentence are obtained using [gensim's built in function](https://tedboy.github.io/nlps/api_gensim.html) which computes cosine similarity between the projection weight vectors of a word and the vectors of each word in the embeddings model.
For example: for input word = 'department,admitted,heart,admitted,emergency'
![newplot(1)](https://user-images.githubusercontent.com/60803118/131270817-65efe5d9-f409-4733-8cea-dd6df3f851cd.png)



## Transformers 

Models used include:[BERT-base](https://github.com/google-research/bert),[Clinical BERT](https://github.com/EmilyAlsentzer/clinicalBERT),[BioMed-RoBERTa](https://huggingface.co/allenai/biomed_roberta_base),[PubMedBERT](https://microsoft.github.io/BLURB/models.html),[MeDAL-Electra](https://github.com/BruceWen120/medal),[Longformer](https://github.com/allenai/longformer) and [TransformerXL](https://github.com/kimiyoung/transformer-xl)

Transformer implementations are based on the open-source PyTorch-transformer repositories [Huggingface](https://github.com/huggingface/transformers) and [Simple Transformers](https://simpletransformers.ai/). 


 
