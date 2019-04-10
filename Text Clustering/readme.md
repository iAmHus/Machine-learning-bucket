## Overview
Implement K-Means algorithm from the scratch - to cluster the data into 7 different classes.

## Scoring Metric
For Evaluation Purposes (Leaderboard Ranking) we will use the V-measure in the sci-kit learn library (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html) that is considered an external index metric to evaluate clustering.


## Data Description

- Input Data (provides as input.dat) consists of 8580 text records in sparse format. 
- No labels are provided.

The format of this file is as follows:
word/feature_id count word/feature_id count .... per line
representing count of words in one document. Mapping of these features_ids to features are available in features.dat

## Overview

- Implement the K-Means Algorithm
- Deal with Text Data (News Records) in Term-Document Sparse Matrix Format.
- Design a Distance Function for Text Data
- Think about Curse of Dimensionality
- Think about Best Metrics for Evaluating Clustering Solutions.