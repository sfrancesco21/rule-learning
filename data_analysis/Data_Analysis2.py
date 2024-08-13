#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:59:42 2024

    -Preprocessing: discarding trials when RT is lower than 0.3 and/or when
    response is missing
    -In test & train - general accuracy, RT, accuracy & RT per feature value
    -Test only: old/new configurations accuracy difference
    -Train only: accuracy when a new triplet is presented (novelty)

@author: Salma Elnagar
"""
 #%%
import scipy.io
import pandas as pd
# import matlab.engine

# specify phase - Test or Training
phase = 'Training'
participants = [98]
df_results = pd.DataFrame()  # create results df to populate later

 #%%
for n, participant_number in enumerate(participants):
    # Load results .mat file
    file_pathway_test = f'/Users/elnagaradmin/ownCloud (2)/Salmas_PhD_NM/Project3/Code/sbj{participant_number}Test.mat'
    file_pathway_training = f'/Users/elnagaradmin/ownCloud (2)/Salmas_PhD_NM/Project3/Code/sbj{participant_number}Training.mat'
    mat_test = scipy.io.loadmat(file_pathway_test)
    mat_training = scipy.io.loadmat(file_pathway_training)

    results_test = mat_test['Results']
    results_training = mat_training['Results']
    df_test = pd.DataFrame(results_test[0][0][0][0])
    df_training = pd.DataFrame(results_training[0][0][0][0])

    # (1) Preprocessing
    for column_training, column_test in zip(df_training.columns, df_test.columns):
        # convert each column to string then cut the two brackets[[]]
        # print(column_training)
        df_training[f"{column_training}"] = df_training[f'{column_training}'].apply(
            lambda x: str(x)).str[2:-2]
        df_test[f"{column_test}"] = df_test[f'{column_test}'].apply(
            lambda x: str(x)).str[2:-2]
        # convert to float
        df_training[f"{column_training}"] = df_training[f"{column_training}"].apply(
            lambda x: float(x))
        df_test[f"{column_test}"] = df_test[f"{column_test}"].apply(
            lambda x: float(x))
        # Disregard trials when RT is lower than 0.3 and that are missed
        df_training = df_training[(df_training['RT'] >= 0.3) & (
            df_training['TrialLost'] == 0)]
        df_test = df_test[(df_test['RT'] >= 0.3) & (df_test['TrialLost'] == 0)]

    # (2) Create results df with desired scores for every participant
    df_results.at[n, 'ppt_number'] = participant_number
    # three scores for both days (6 columns)
    results = ['Feedback', 'Action', 'RT', ]
    for result in results:
        df_results.at[n, f'{result}_training'] = df_training[result].mean()
        df_results.at[n, f'{result}_test'] = df_test[result].mean()
    # Old/new accuracy the mean of the outcome for the configurations in test
    # that are (old) and are not (new) in configuration in training
    train_configurations = set(df_training['Configuration'])
    # Calculate the mean of the outcome for test rows where the configuration is present in the training set (old)
    df_results.at[n, 'old_accuracy'] = df_test[df_test['Configuration'].isin(
        train_configurations)]['Feedback'].mean()
    # Calculate the mean of the outcome for test rows where the configuration is not present in the training set (new)
    df_results.at[n, 'new_accuracy'] = df_test[~df_test['Configuration'].isin(
        train_configurations)]['Feedback'].mean()
    # Old and new configurations accuracy difference
    df_results.at[n, 'oldnew_acccuracy_difference'] = df_results.at[n,
                                                                    'old_accuracy'] - df_results.at[n, 'new_accuracy']

    # (3) Get RT and accuracy per feature value
    # 3.1. Create grid: define the range of values for x and y
    y_values = range(1, 7)
    x_values = range(1, 7)

    # Initialize an empty list to store the grid information
    grid_data = []

    # Generate the grid and combination numbers
    combination_number = 0
    for y in y_values:
        for x in x_values:
            combination_number += 1
            grid_data.append((y, x, combination_number))

    # Create a DataFrame from the grid data
    df_feature_value = pd.DataFrame(
        grid_data, columns=['y', 'x', 'combination_number'])

    # Get the diagonal index
    df_feature_value['feature_value'] = (
        df_feature_value['x'] - df_feature_value['y']).abs()

    # Merge two dfs so that we map the configurations in our current dfs to the right feature values
    df_feature_value_selected = df_feature_value[[
        'combination_number', 'feature_value']]

    df_test = pd.merge(df_test, df_feature_value_selected,
                       left_on='Configuration', right_on='combination_number', how='left')
    df_test = df_test.drop(columns=['combination_number'])

    df_training = pd.merge(df_training, df_feature_value_selected,
                           left_on='Configuration', right_on='combination_number', how='left')
    df_training = df_training.drop(columns=['combination_number'])

    # 3.2. Get the outcome and RT of the different feature values and put it in results
    features = list(range(0, 6))
    for feature in features:
        # Get mean accuracy per feature
        df_results[f'Accuracy_FV{feature}'] = df_test.query(
            f'feature_value == {feature}')['Feedback'].mean()
        # Get mean RT per feature
        df_results[f'RT_FV{feature}'] = df_test.query(
            f'feature_value == {feature}')['RT'].mean()

    # (4) In the training phase (blocked design only) get the accuracy when
    # new triplets are presented. Skip the first ever triplet to be presented.
    # get accuracy only for when a triplet is the first time.

    # Define the segment size
    segment_size = 30

    # Initialize list to store segment means
    segment_means = []

    # Loop over segments, skipping the first 3 trials in the first segment
    for s, start_idx in enumerate(range(segment_size, len(df_training), segment_size)):
        # Get the first 3 rows of the current segment
        segment_rows = df_training.iloc[start_idx:start_idx + 3]

        # Calculate the mean of the 'feeback' column for these rows
        df_results[f'Accuracy_Triplet{s+1}'] = segment_rows['Feedback'].mean()
