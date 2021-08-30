### Various options for training fastText embeddings where parameter choices are varied.

./fasttext cbow -input health-related-text-input.txt -output T50 -dim 50 -ws 5 -neg 10 -minn 5 -maxn 5; # option I CBOW

./fasttext cbow -input health-related-text-input.txt -output TREC3ws50 -dim 50 -ws 3 -neg 10 -minn 5 -maxn 5;  # option II CBOW

./fasttext cbow -input health-related-text-input.txt -output TREC7ws50 -dim 50 -ws 7 -neg 10 -minn 5 -maxn 5;  # option III CBOW

./fasttext cbow -input health-related-text-input.txt -output TREC5neg50 -dim 50 -ws 5 -neg 5 -minn 5 -maxn 5;  # option IV CBOW

./fasttext cbow -input health-related-text-input.txt -output TRECdefminn50 -dim 50 -ws 5 -neg 10;  # option V CBOW

./fasttext cbow -input health-related-text-input.txt -output TRECmin3max50 -dim 50 -ws 5 -neg 10 -minn 3 -maxn 3; # option VI CBOW

./fasttext cbow -input health-related-text-input.txt -output TREChs50 -dim 50 -ws 5 -neg 10 -minn 5 -maxn 5 -loss hs  # option VII CBOW

./fasttext cbow -input health-related-text-input.txt -output TRECnsam50 -dim 50 -ws 5 -neg 10 -minn 5 -maxn 5 -loss ns;  # option VIII CBOW

./fasttext cbow -input health-related-text-input.txt -output TREC10epoh50 -dim 50 -ws 5 -neg 10 -minn 5 -maxn 5 -epoch 10;  # option IX CBOW

./fasttext skipgram -input health-related-text-input.txt -output T50SG -dim 50 -ws 5 -neg 10 -minn 5 -maxn 5; # option I Skip-gram
