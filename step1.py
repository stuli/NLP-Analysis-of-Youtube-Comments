import pandas as pd
import numpy as np
import sklearn as sk
from os import system

import constants


name = 'step1'

# Define functions on the comment and creator_name columns for aggregation


def combine_comments(comment):
    return ' ; '.join(comment)


def avg_length(comment):
    return len(combine_comments(comment))//comment.count()


def avg_nemojis(comment):
    # common emojis
    return sum(comment.str.count(r'[\u263a-\U0001f645]'))//comment.count()


def avg_npunctuations(comment):
    # common punctuations
    return sum(comment.str.count(r'[?!.,]'))//comment.count()


def avg_ntags(comment):
    # @ and # tags
    return sum(comment.str.count(r'[@#]'))//comment.count()  


def uses_dog_keyword(comment):
    return any(comment.str.contains('|'.join(constants.dog_filter_list), case=False))


def uses_cat_keyword(comment):
    return any(comment.str.contains('|'.join(constants.cat_filter_list), case=False))


def all_creators(creator_name):
    return ' ; '.join(creator_name)


def contains_dog_keyword(creator_name):
    return any(creator_name.str.contains('|'.join(constants.dog_filter_list), case=False))


def contains_cat_keyword(creator_name):
    return any(creator_name.str.contains('|'.join(constants.cat_filter_list), case=False))


def step1():
    df = pd.read_csv('../animals_comments.csv', header='infer')
    df = df.dropna()
    df.creator_name = df.creator_name.astype(str)

    # Aggregate the data by userid, using the following built-in and defined functions:
    #df = df.sample(100000) # Use for running on a smaller sample
    aggregated_df = pd.DataFrame(df.groupby('userid').agg(
        {
            'comment': [combine_comments, 'count', avg_length, avg_nemojis, avg_npunctuations,
                        avg_ntags, uses_dog_keyword, uses_cat_keyword],
            'creator_name': [all_creators, 'nunique', contains_dog_keyword, contains_cat_keyword]
        }
    ))

    # label = [0, 1, 2, -1] if user is [cat, dog, both cat and dog, neither cat nor dog] owner.
    func = aggregated_df[('comment', 'combine_comments')].str.contains
    aggregated_df['label'] = 2 * func(constants.phrases_dog_ownership, case=False) + \
        func(constants.phrases_cat_ownership, case=False) - 1

    labelled_df = aggregated_df.loc[aggregated_df.label.isin([0, 1])]

    labelled_df.to_csv(f'{name}_labels.csv', header=True)
    aggregated_df.to_pickle(f'{name}_aggregated.pkl')
    return aggregated_df, labelled_df


if __name__ == "__main__":
    system(f'mkdir {name}_output')
    step1()
    system(f'mv {name}_*.csv {name}_*.pkl {name}_output/')
