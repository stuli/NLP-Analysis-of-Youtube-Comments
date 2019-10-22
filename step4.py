import step2
import pandas as pd
import numpy as np
import sklearn as sk
from os import system


name = 'step4'

try:
    trained_methods = step2.Step2Results.deserialize()
except FileNotFoundError as e:
    print(f"Error {e}: re-running step 2")
    trained_methods = step2.step2()


if __name__ == "__main__":
    feature_list_tfidf = list(trained_methods.featurized_training_data.columns)

    # Obtain the 5 features that the classifier weighted most heavily
    important_features = trained_methods.trained_classifier.coef_.argsort()

    f = open('%s.txt' % (name), 'w+')

    f.write('The top 5 features that differentiate cat and dog owners are: \n')

    for i in important_features[0][:5]:
        # They can be words from the comments
        if i < 2000:
            f.write(f'The relative frequency of \'{trained_methods.vectorizer.get_feature_names()[i]}\' in their comments. \n')
        # They can also be the constructed features
        else:
            f.write(f'The following attribute of their commenting pattern: {feature_list_tfidf[i-1]}. \n')

    f.close()

    system(f'mkdir {name}_output')
    system(f'mv {name}.txt {name}_output/')
