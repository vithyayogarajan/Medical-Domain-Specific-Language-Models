# Health-Related Document Level Embeddings


## Training Embeddings 
Embeddings presented in the research are pre-trained:
- using [fastText](https://fasttext.cc/) 
- at document level
- for both [CBOW](http://dblp.org/rec/bib/journals/corr/abs-1301-3781) (T300 and T600) and [Skip-gram](http://dblp.org/rec/bib/journals/corr/abs-1301-3781) (T300SG, T600SG) models


## Training Data  

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



## References
When using our pre-trained models for your application, please cite the following papers:
1.  Yogarajan,  V.,  Gouk,  H.,  Smith,  T.,  Mayo,  M.,  Pfahringer,  B.:  Comparing  High  Dimensional Word Embeddings Trained on Medical Text to Bag-of-Words For Predicting Medical  Codes.   Proceedings  of  the  Asian  Conference  on  Intelligent  Information  andDatabase  Systems  (ACIIDS  2020).  In  N.  T.  Nguyen  et  al.  (Eds.),  Lecture  Notes  on Artificial Intelligence (LNAI), Springer Nature. 12033, 1–12 (2020)

2. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2020). Seeing The Whole Patient: Using Multi-Label Medical Text Classification Techniques to Enhance Predictions of Medical Codes. [arXiv preprint arXiv:2004.00430](https://arxiv.org/abs/2004.00430). 


