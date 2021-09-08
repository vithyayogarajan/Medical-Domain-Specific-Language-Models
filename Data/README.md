# Data

[MIMIC-III](https://physionet.org/works/MIMICIIIClinicalDatabase/access.shtml) and [eICU](https://eicu-crd.mit.edu/) needs access and can be obtained from [PhysioNet](https://physionet.org/).

Sample data is provided here to provide an understanding of the structure of the data used for experiments.

 - For binary classification: .csv files for MIMIC-III include:
 
|HADM_ID|Text|Labels|
|:-----|:----|----:|
|145834|pre-trained discharge summary for patient with #HADM_ID = 145834, MIMIC-III data|1|

 - For multi-label classification with 'L' number of labels: .csv files for MIMIC-III include:
 
|HADM_ID|Text|Label_1|Label_2|...|Label_L|
|:-----|:----|----:|----:|----:|----:|
|145834|pre-trained discharge summary for patient with #HADM_ID = 145834, MIMIC-III data|0 |1 |...|0|

 - Arff files for WEKA and MEKA includes static features obtained using pre-trained embeddings and class labels. See [embeddings processing](https://github.com/vithyayogarajan/Medical-Domain-Specific-Language-Models/blob/main/Data_label_processing/data_text_to_embeddings.ipynb) for more details.   

- eICU data uses 'pid' (patient i.d.) as the key for database, instead of 'HADM_ID' (Hospital admission i.d.) in MIMIC-III.

- [Emb50.txt](https://drive.google.com/file/d/1rcApyx9uh9_anUpESd9uQYF8sJ7SDivH/view?usp=sharing) is a sample pre-trained embeddings with 50 dimension (CBOW model).




