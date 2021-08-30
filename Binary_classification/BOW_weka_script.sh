java -cp weka-3-8-0-monolithic.jar -Xmx16G weka.filters.unsupervised.attribute.StringToWordVector -W 100000 -C -T -I -M 10 -N 1 -i text.arff> text100000.arff;

java -cp weka-3-8-0-monolithic.jar -Xmx16G weka.classifiers.bayes.NaiveBayesMultinomial -t text100000.arff -c 1 > BOW/output/bloodtextbow100000.txt ;


