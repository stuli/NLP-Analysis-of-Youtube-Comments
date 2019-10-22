import step2
import step3
import pandas as pd
import numpy as np
import sklearn as sk
import pickle
from os import system


name = 'step5'

try:
    trained_methods = step2.Step2Results.deserialize()
except FileNotFoundError as e:
    print(f"Error {e}: re-running step 2")
    trained_methods = step2.step2()

try:
    with open('step3_output/step3_predicted_users.pkl', 'rb') as handle:
        predicted_users = pickle.load(handle)
except FileNotFoundError as e:
    print(f"Error {e}: re-running step 3")
    predicted_users = step3.step3()

# Define the following function to determine the fraction of users commenting on each channel
# who are cat/dog owners, and the error in this calculation


def frac_ownership(userid):
    dog_owners = 0
    error_dog_owners = 0
    cat_owners = 0
    error_cat_owners = 0
    nunique_users = userid.nunique()
    for user in userid.unique():
        if predicted_users[user] == 1:
            dog_owners += 1
        elif predicted_users[user] == 0:
            cat_owners += 1
    return [
        round(dog_owners / nunique_users, 2),
        round(dog_owners * (trained_methods.fpr_dog_owner) / nunique_users, 2),
        round(cat_owners / nunique_users, 2),
        round(cat_owners * (trained_methods.fpr_cat_owner) / nunique_users, 2)
    ]


if __name__ == "__main__":
    df = pd.read_csv('../animals_comments.csv', header='infer')
    df = df.dropna()
    df.creator_name = df.creator_name.astype(str)

    # Group the data by the name of the creator, using the above function and built in
    # functions for aggregation
    creator_aggregated_df = pd.DataFrame(df.groupby('creator_name').agg(
        {
            'userid': ['count', 'unique', 'nunique', frac_ownership]
        }
    ))

    # Make selections to keep only data for content creators who have enough
    # absolute and unique user comments
    reliable_creators_df = creator_aggregated_df.loc[
        (creator_aggregated_df[('userid', 'count')] > 50)
        & ((creator_aggregated_df[('userid', 'nunique')] /
            creator_aggregated_df[('userid', 'count')]) > 0.8)
    ]

    reliable_creators_df = pd.DataFrame(reliable_creators_df.userid.frac_ownership.tolist(),
                                        columns=['frac_dog_owners', 'error_frac_dog_owner',
                                                 'frac_cat_owners', 'error_frac_cat_owner'],
                                        index=reliable_creators_df.index
                                        )

    # Sort the data to obtain the top 10 creators with largest cat/dog onership auciences
    creators_large_cat_owner_audience = reliable_creators_df.sort_values(
        'frac_cat_owners',
        ascending=False).head(10).drop(columns=['frac_dog_owners', 'error_frac_dog_owner'])

    creators_large_dog_owner_audience = reliable_creators_df.sort_values(
        'frac_dog_owners',
        ascending=False).head(10).drop(columns=['frac_cat_owners', 'error_frac_cat_owner'])

    pd.concat([creators_large_cat_owner_audience,
               creators_large_dog_owner_audience]).to_csv(f'{name}.csv')

    system(f'mkdir {name}_output')

    system(f'mv {name}.csv {name}_output/')
