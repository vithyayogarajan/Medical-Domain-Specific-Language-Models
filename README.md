# Medical-Domain-Specific-Language-Models

This repositary contains code used for following the PhD thesis and publications:

1. Yogarajan, V (2021). Domain-specific Language Models for Multi-label Classification of Medical Text. The University of Waikato. PhD Thesis. (examination process)
2. Yogarajan, V., Montiel J., Smith T., & Pfahringer B. (2021) Transformers for Multi-label Classification of Medical Text: An Empirical Comparison. In: Tucker A., Henriques Abreu P., Cardoso J., Pereira Rodrigues P., Riaño D. (eds) Artificial Intelligence in Medicine. AIME 2021. Lecture Notes in Computer Science, vol 12721. Springer, Cham. [link](https://doi.org/10.1007/978-3-030-77211-6_12)
3. Yogarajan, V., Gouk, H., Smith, T., Mayo, M., & Pfahringer, B. (2020). Comparing High Dimensional Word Embeddings Trained on Medical Text to Bag-of-Words for Predicting Medical Codes. In Asian Conference on Intelligent Information and Database Systems. Springer, Cham, pp. 97-108.
4. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2020). Seeing The Whole Patient: Using Multi-Label Medical Text Classification Techniques to Enhance Predictions of Medical Codes. [arXiv preprint arXiv:2004.00430](https://arxiv.org/abs/2004.00430). 
5.Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Concatenated BioMed-Transformers for Multi-label Classification of Medical Text. (under submission)
6. Yogarajan, V., Montiel, J., Smith, T., & Pfahringer, B. (2021). Predicting COVID-19 Patient Shielding: A Comprehensive Study. (under submission).
 

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

 
