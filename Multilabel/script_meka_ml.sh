java -Xmx16G -cp "./lib/*" meka.classifiers.multilabel.BR -t input_multilabel_18labels.arff -x 10 -verbosity 4 -W weka.classifiers.functions.Logistic -- -R 1  > output.txt ;

java -Xmx16G -cp "./lib/*" meka.classifiers.multilabel.meta.BaggingML -W meka.classifiers.multilabel.CC -verbosity 4 -t input_multilabel_18labels.arff -- -S 0 -W weka.classifiers.functions.Logistic -- -R 1.0 > output.txt ;

