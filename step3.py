import step1
import step2
import constants
import pandas as pd
import numpy as np
import sklearn as sk
import pickle
from os import system
pd.options.mode.chained_assignment = None


name = 'step3'

try:
    aggregated_df = pd.read_pickle('step1_output/step1_aggregated.pkl')
except (FileNotFoundError, BaseException) as e:
    print(f"Error {e}: re-running step 1")
    aggregated_df, _ = step1.step1()

try:
    trained_methods = step2.Step2Results.deserialize()
except FileNotFoundError as e:
    print(f"Error {e}: re-running step 2")
    trained_methods = step2.step2()


def step3():
    # Process all the data in the same way as the training and test sets
    constructed_features_all_data = aggregated_df[constants.feature_list]
    vectorized_comments_all_data = trained_methods.vectorizer.transform(
        aggregated_df.comment.combine_comments)
    vectorized_comments_all_data_tfidf = trained_methods.tfidf_transformer.transform(
        vectorized_comments_all_data)
    vectorized_comments_all_data_tfidf_df = pd.DataFrame(
        vectorized_comments_all_data_tfidf.todense())
    vectorized_comments_all_data_tfidf_df.columns = pd.MultiIndex.from_product(
        [['comment_text'], vectorized_comments_all_data_tfidf_df.columns])

    # Standardize the count features since we are using an SVM classifier:
    constructed_features_all_data[constants.features_to_scale] = \
        trained_methods.scaler.transform(
            constructed_features_all_data[constants.features_to_scale])

    # Join all the features together
    rows, columns = aggregated_df.shape
    constructed_features_all_data['row_number'] = [l for l in range(rows)]

    featurized_all_data = pd.merge(vectorized_comments_all_data_tfidf_df,
                                   constructed_features_all_data,
                                   left_on=vectorized_comments_all_data_tfidf_df.index,
                                   right_on=constructed_features_all_data.row_number)
    featurized_all_data = featurized_all_data.drop(
        columns=[('row_number', ''), 'key_0'])

    # Predict cat and dog owners amongst all the users using the chosen classifier
    prediction_all_data = trained_methods.trained_classifier.predict(
        featurized_all_data)

    aggregated_df['predicted_label'] = prediction_all_data
    predicted_users = aggregated_df['predicted_label'].to_dict() 

    # Count number of predicted cat/dog owners
    npredicted_dog_owners = aggregated_df.loc[aggregated_df.predicted_label == 1, 'predicted_label'].sum(
    )
    npredicted_cat_owners = len(prediction_all_data) - npredicted_dog_owners

    # Determine actual number of cat/dog owners correcting for false positive rates
    nactual_dog_owners = int(npredicted_dog_owners *
                             (1 - trained_methods.fpr_dog_owner))
    nactual_cat_owners = int(npredicted_cat_owners *
                             (1 - trained_methods.fpr_cat_owner))

    # Compute fraction of all users that are cat/dog owners
    frac_dog_owners = nactual_dog_owners/len(aggregated_df)
    frac_cat_owners = nactual_cat_owners/len(aggregated_df)

    aggregated_df.predicted_label.to_csv('%s.csv' % (name), header=True)

    with open('%s.txt' % (name), 'a') as f:
        f.write('Fraction of all users who are cat owners: %.2f' %
                (frac_cat_owners))
        f.write('Fraction of all users who are dog owners: %.2f' %
                (frac_dog_owners))
    return predicted_users


if __name__ == "__main__":
    system(f'mkdir {name}_output')
    predicted_users = step3()
    with open(f'{name}_predicted_users.pkl', 'wb') as handle:
        pickle.dump(predicted_users, handle)
    system(f'mv {name}.txt {name}.csv {name}_predicted_users.pkl {name}_output/')
