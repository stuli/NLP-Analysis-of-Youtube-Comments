import step1
import constants
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import sklearn as sk
from os import system
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


name = 'step2'

try:
    labelled_df = pd.read_csv('step1_output/step1_labels.csv',
                              header=[0, 1], index_col=0)
except FileNotFoundError as e:
    print(f"Error {e}: re-running step 1")
    _, labelled_df = step1.step1()


class Step2Results(object):
    __slots__ = [
        "vectorizer",
        "tfidf_transformer",
        "trained_classifier",
        "fpr_dog_owner",
        "fpr_cat_owner",
        "scaler",
        "featurized_training_data"
    ]

    def __init__(self,
                 vectorizer=None,
                 tfidf_transformer=None,
                 trained_classifier=None,
                 fpr_dog_owner=None,
                 fpr_cat_owner=None,
                 scaler=None,
                 featurized_training_data=None
                 ):
        self.vectorizer = vectorizer
        self.tfidf_transformer = tfidf_transformer
        self.trained_classifier = trained_classifier
        self.fpr_dog_owner = fpr_dog_owner
        self.fpr_cat_owner = fpr_cat_owner
        self.scaler = scaler
        self.featurized_training_data = featurized_training_data

    def serialize(self):
        path = f"{name}_results_" + "{attribute}.pkl"
        for attribute_name in self.__slots__:
            attribute = getattr(self, attribute_name)
            joblib.dump(
                attribute,
                path.format(attribute=attribute_name)
            )

    @classmethod
    def deserialize(cls):
        path = f"{name}_output/{name}_results_" + "{attribute}.pkl"
        init_dict = dict()
        for attribute_name in cls.__slots__:
            attribute = joblib.load(path.format(attribute=attribute_name))
            init_dict[attribute_name] = attribute
        return cls(**init_dict)


def step2():
    # Split the labelled data into train and test portions
    data_train, data_test, label_train, label_test = \
        train_test_split(labelled_df, labelled_df.label, test_size=0.2)

    ### First, we try classifying using our constructed features only ###
    # Keep only count and binary type features
    constructed_features_train = data_train[constants.feature_list]
    constructed_features_test = data_test[constants.feature_list]

    ## Multinomial Naive Bayes Classifier ##
    classifier_MNB = MultinomialNB()

    # Train and predict
    trained_classifier_MNB = classifier_MNB.fit(
        constructed_features_train, label_train)
    prediction_MNB = trained_classifier_MNB.predict(constructed_features_test)
    score_MNB = trained_classifier_MNB.score(
        constructed_features_test, label_test)
    label_prediction_probability_MNB = trained_classifier_MNB.predict_proba(
        constructed_features_test)

    # Get ROC curve and compute AUC
    false_positive_rate_MNB, true_positive_rate_MNB, _ = roc_curve(
        label_test,
        label_prediction_probability_MNB[:, 0], pos_label=0)
    roc_auc_MNB = auc(false_positive_rate_MNB, true_positive_rate_MNB)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(
        false_positive_rate_MNB,
        true_positive_rate_MNB,
        color='darkgreen',
        lw=lw,
        label='ROC curve (Area under ROC curve = %0.2f)' % roc_auc_MNB)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for cat and dog'
              '\nownership prediction using constructed features')
    plt.legend(
        loc="lower right",
        fontsize='medium',
        title='Classifier: Multinomial Naive Bayes\nScore = %0.2f' % (score_MNB))
    plt.savefig(f'{name}_ROC_MNB.pdf', bbox_inches='tight')

    # Obtain the false positive rates for predicting dog and cat ownership using the confusion matrix
    cm_MNB = confusion_matrix(label_test, prediction_MNB)
    tn, fp, fn, tp = cm_MNB.ravel()
    fpr_dog_owner_MNB = fp / (fp + tn)
    fpr_cat_owner_MNB = fn / (fn + tp)

    ## Support Vector Machine Classifier ##
    classifier_SVM = SGDClassifier(loss='hinge', penalty='l2', max_iter=100)

    # Since this is a distance-based classifier, standardize the count type features
    scaler = StandardScaler()
    constructed_features_train_scaled = constructed_features_train.copy()
    constructed_features_test_scaled = constructed_features_test.copy()
    constructed_features_train_scaled[constants.features_to_scale] = \
        scaler.fit_transform(
            constructed_features_train[constants.features_to_scale])
    constructed_features_test_scaled[constants.features_to_scale] = \
        scaler.transform(
            constructed_features_test[constants.features_to_scale])

    # Train and predict
    trained_classifier_SVM = classifier_SVM.fit(
        constructed_features_train_scaled, label_train)
    prediction_SVM = trained_classifier_SVM.predict(
        constructed_features_test_scaled)
    score_SVM = trained_classifier_SVM.score(
        constructed_features_test_scaled, label_test)
    label_prediction_probability_SVM = \
        trained_classifier_SVM.decision_function(
            constructed_features_test_scaled)

    # Get ROC curve and compute AUC
    false_positive_rate_SVM = dict()
    true_positive_rate_SVM = dict()
    roc_auc_SVM = dict()
    false_positive_rate_SVM, true_positive_rate_SVM, _ = \
        roc_curve(label_test, label_prediction_probability_SVM)
    roc_auc_SVM = auc(false_positive_rate_SVM, true_positive_rate_SVM)

    # Plot ROC curve
    plt.figure()
    plt.plot(false_positive_rate_SVM, true_positive_rate_SVM, color='darkgreen',
             lw=lw, label='ROC curve (Area under ROC curve = %0.2f)' % roc_auc_SVM)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for cat and dog\nownership prediction using constructed features')
    plt.legend(loc="lower right", fontsize='medium',
               title='Classifier: Support Vector Machine\nScore = %0.2f' % (score_SVM))
    plt.savefig(f'{name}_ROC_SVM.pdf', bbox_inches='tight')

    # Obtain false positve rates
    cm_SVM = confusion_matrix(label_test, prediction_SVM)
    tn, fp, fn, tp = cm_SVM.ravel()
    fpr_dog_owner_SVM = fp/(fp+tn)
    fpr_cat_owner_SVM = fn/(fn+tp)

    ### Next, we incorporate the text of the comments as well ###
    # Exclude the phrases used to determine ownership labels to avoid biasing the classifier
    stopwords = text.ENGLISH_STOP_WORDS.union(constants.dog_stopwords)
    stopwords = stopwords.union(constants.cat_stopwords)

    # Featurize the comments using bag of words and term frequency-inverse document frequency
    vectorizer = CountVectorizer(stop_words=stopwords, lowercase=True,
                                 max_features=2000, analyzer='word',
                                 token_pattern=r'(?u)\b\w\w+\b|[\u263a-\U0001f645]+?')
    tfidf_transformer = TfidfTransformer()

    # Fit and transform the train data
    vectorized_comments_train = vectorizer.fit_transform(
        data_train.comment.combine_comments)
    vectorized_comments_train_tfidf = tfidf_transformer.fit_transform(
        vectorized_comments_train)
    vectorized_comments_train_tfidf_df = pd.DataFrame(
        vectorized_comments_train_tfidf.todense())
    vectorized_comments_train_tfidf_df.columns = pd.MultiIndex.from_product(
        [['comment_text'], vectorized_comments_train_tfidf_df.columns])

    # Add the features from comments to the constructed features
    train_rows, train_columns = data_train.shape
    constructed_features_train['row_number'] = [l for l in range(train_rows)]
    featurized_training_data = pd.merge(vectorized_comments_train_tfidf_df,
                                        constructed_features_train,
                                        left_on=vectorized_comments_train_tfidf_df.index,
                                        right_on=constructed_features_train.row_number)
    featurized_training_data = featurized_training_data.drop(
        columns=[('row_number', ''), 'key_0'])

    # Transform and featurize the test data in the same way
    vectorized_comments_test = vectorizer.transform(
        data_test.comment.combine_comments)
    vectorized_comments_test_tfidf = tfidf_transformer.transform(
        vectorized_comments_test)
    vectorized_comments_test_tfidf_df = pd.DataFrame(
        vectorized_comments_test_tfidf.todense())
    vectorized_comments_test_tfidf_df.columns = pd.MultiIndex.from_product(
        [['comment_text'], vectorized_comments_test_tfidf_df.columns])

    test_rows, test_columns = data_test.shape
    constructed_features_test['row_number'] = [l for l in range(test_rows)]
    featurized_test_data = pd.merge(vectorized_comments_test_tfidf_df,
                                    constructed_features_test,
                                    left_on=vectorized_comments_test_tfidf_df.index,
                                    right_on=constructed_features_test.row_number)
    featurized_test_data = featurized_test_data.drop(
        columns=[('row_number', ''), 'key_0'])

    ## Multinomial Naive Bayes Classifier, including comments ##
    classifier_tfidf_MNB = MultinomialNB()

    # Train and predict
    trained_classifier_tfidf_MNB = classifier_tfidf_MNB.fit(
        featurized_training_data, label_train)
    prediction_tfidf_MNB = trained_classifier_tfidf_MNB.predict(
        featurized_test_data)
    score_tfidf_MNB = trained_classifier_tfidf_MNB.score(
        featurized_test_data, label_test)
    label_prediction_probability_tfidf_MNB = \
        trained_classifier_tfidf_MNB.predict_proba(featurized_test_data)

    # Get ROC curve and compute AUC
    false_positive_rate_tfidf_MNB = dict()
    true_positive_rate_tfidf_MNB = dict()
    roc_auc_tfidf_MNB = dict()
    false_positive_rate_tfidf_MNB, true_positive_rate_tfidf_MNB, _ = \
        roc_curve(
            label_test, label_prediction_probability_tfidf_MNB[:, 0], pos_label=0)
    roc_auc_tfidf_MNB = auc(false_positive_rate_tfidf_MNB,
                            true_positive_rate_tfidf_MNB)

    # Plot ROC curve
    plt.figure()
    plt.plot(false_positive_rate_tfidf_MNB, true_positive_rate_tfidf_MNB, color='darkgreen',
             lw=lw, label='ROC curve (Area under ROC curve = %0.2f)' % roc_auc_tfidf_MNB)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for cat and dog\nownership prediction including the text of comments')
    plt.legend(loc="lower right", fontsize='medium',
               title='Classifier: Multinomial Naive Bayes\nScore = %0.2f' % (score_tfidf_MNB))
    plt.savefig(f'{name}_ROC_tfidf_MNB.pdf', bbox_inches='tight')

    # Obtain false positive rates
    cm_tfidf_MNB = confusion_matrix(label_test, prediction_tfidf_MNB)
    tn, fp, fn, tp = cm_tfidf_MNB.ravel()
    fpr_dog_owner_tfidf_MNB = fp/(fp+tn)
    fpr_cat_owner_tfidf_MNB = fn/(fn+tp)

    ## Support Vector Machine Classifier, including comments ##
    classifier_tfidf_SVM = SGDClassifier(
        loss='hinge', penalty='l2', max_iter=100)

    # This time, add comment features to the *standardized* train and test data
    constructed_features_train_scaled['row_number'] = [
        l for l in range(train_rows)]
    featurized_training_data_scaled = pd.merge(vectorized_comments_train_tfidf_df,
                                               constructed_features_train_scaled,
                                               left_on=vectorized_comments_train_tfidf_df.index,
                                               right_on=constructed_features_train_scaled.row_number)
    featurized_training_data_scaled = featurized_training_data_scaled.drop(
        columns=[('row_number', ''), 'key_0'])

    constructed_features_test_scaled['row_number'] = [
        l for l in range(test_rows)]
    featurized_test_data_scaled = pd.merge(vectorized_comments_test_tfidf_df,
                                           constructed_features_test_scaled,
                                           left_on=vectorized_comments_test_tfidf_df.index,
                                           right_on=constructed_features_test_scaled.row_number)
    featurized_test_data_scaled = featurized_test_data_scaled.drop(
        columns=[('row_number', ''), 'key_0'])

    # Train and predict
    trained_classifier_tfidf_SVM = classifier_tfidf_SVM.fit(
        featurized_training_data_scaled, label_train)
    prediction_tfidf_SVM = trained_classifier_tfidf_SVM.predict(
        featurized_test_data_scaled)
    score_tfidf_SVM = trained_classifier_tfidf_SVM.score(
        featurized_test_data_scaled, label_test)
    label_prediction_probability_tfidf_SVM = \
        trained_classifier_tfidf_SVM.decision_function(
            featurized_test_data_scaled)

    # Get ROC and compute AUC
    false_positive_rate_tfidf_SVM, true_positive_rate_tfidf_SVM, _ = \
        roc_curve(label_test, label_prediction_probability_tfidf_SVM)
    roc_auc_tfidf_SVM = auc(false_positive_rate_tfidf_SVM,
                            true_positive_rate_tfidf_SVM)

    # Plot ROC curve
    plt.figure()
    plt.plot(false_positive_rate_tfidf_SVM, true_positive_rate_tfidf_SVM, color='darkgreen',
             lw=lw, label='ROC curve (Area under ROC curve = %0.2f)' % roc_auc_tfidf_SVM)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve for cat and dog\nownership prediction including the text of comments')
    plt.legend(loc="lower right", fontsize='medium',
               title='Classifier: Support Vector Machine\nScore = %0.2f' % (score_tfidf_SVM))
    plt.savefig(f'{name}_ROC_tfidf_SVM.pdf', bbox_inches='tight')

    # Obtain false positive rates
    cm_tfidf_SVM = confusion_matrix(label_test, prediction_tfidf_SVM)
    tn, fp, fn, tp = cm_tfidf_SVM.ravel()
    fpr_dog_owner_tfidf_SVM = fp/(fp+tn)
    fpr_cat_owner_tfidf_SVM = fn/(fn+tp)

    # Based on the results the SVM classifier including comment text features is chosen to classify the data
    trained_classifier = trained_classifier_tfidf_SVM
    fpr_dog_owner = fpr_dog_owner_tfidf_SVM
    fpr_cat_owner = fpr_cat_owner_tfidf_SVM

    output = Step2Results(
        vectorizer,
        tfidf_transformer,
        trained_classifier,
        fpr_dog_owner,
        fpr_cat_owner,
        scaler,
        featurized_training_data
    )
    output.serialize()

    return output


if __name__ == "__main__":
    system(f'mkdir {name}_output')
    step2()
    system(f'mv {name}*.pdf {name}_output/')
    system(f'mv {name}*.pkl {name}_output/')
