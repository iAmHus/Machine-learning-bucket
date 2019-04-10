# %%

import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# %%
training_data = pd.read_csv('data/traindrugs.txt', sep='	', header=None)
vectorizer = CountVectorizer(
    tokenizer=None,
    preprocessor=None,
    stop_words=None,
    max_features=120000,
    token_pattern=r"(?u)\b\w+\b")

corpus = []

for x in range(1, 100001, 1):
    corpus.append(str(x))

X = vectorizer.fit([' '.join(corpus)])

output = vectorizer.transform(training_data.loc[:, 1])

features_df = pd.DataFrame(output.todense(), columns=vectorizer.get_feature_names())
target_df = training_data.loc[:, 0]

target_df.value_counts()

features_df.shape
# features_df.loc[0, '62214']


# %%
# Read test data
test_data = pd.read_csv('data/testdrugs.txt', sep=',', header=None)

# test_data.tail(1)
test_data.columns

test_output = vectorizer.transform(test_data.loc[:, 0])

test_features_df = pd.DataFrame(test_output.todense(), columns=vectorizer.get_feature_names())

# %%

# Combining features in train and test dataframes
combined_features_df = pd.concat([features_df, test_features_df], keys=['x', 'y'])

features_df.shape
test_features_df.shape

combined_features_df.shape
combined_features_df.loc['y']
# %%
# 3.b PCA
# PCA for dimensionality reduction

scaler = StandardScaler()

scaler.fit(combined_features_df)

# scaled_combined_data = scaler.transform(features_df)
scaled_combined_data = scaler.transform(combined_features_df)

pca = PCA(.90)
pca.fit(scaled_combined_data)

# pca.n_components_
pca_combined_df = pd.DataFrame(pca.transform(scaled_combined_data))

pca_test_df = pca_combined_df.iloc[800:]
pca_train_df = pca_combined_df.iloc[:800]

X_train, X_test, y_train, y_test = train_test_split(pca_train_df, target_df, train_size=.8)

# %%
# from imblearn.combine import SMOTEENN
#
# sm = SMOTEENN()
#
# X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
#
# X_train = pd.DataFrame(X_resampled)
# y_train = pd.Series(y_resampled)
# y_train.value_counts()

# GradientBoostingClassifier(criterion='friedman_mse', init=None,
#                            learning_rate=0.25, loss='deviance', max_depth=3,
#                            max_features=None, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_samples_split=2,
#                            min_weight_fraction_leaf=0.0, n_estimators=200,
#                            n_iter_no_change=None, presort='auto', random_state=0,
#                            subsample=1.0, tol=0.0001, validation_fraction=0.1,
#                            verbose=0, warm_start=False)
# 0.5333333333333334

# print('Resampled dataset shape {}'.format(Counter(y_resampled)))
# %%
# SMOTE TOMEK

#
# from imblearn.combine import SMOTETomek
#
# sm = SMOTETomek()
#
# X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
#
# X_train = pd.DataFrame(X_resampled)
# y_train = pd.Series(y_resampled)
# y_train.value_counts()
# #
# # GradientBoostingClassifier(criterion='friedman_mse', init=None,
# #                            learning_rate=0.25, loss='deviance', max_depth=3,
# #                            max_features=None, max_leaf_nodes=None,
# #                            min_impurity_decrease=0.0, min_impurity_split=None,
# #                            min_samples_leaf=1, min_samples_split=2,
# #                            min_weight_fraction_leaf=0.0, n_estimators=200,
# #                            n_iter_no_change=None, presort='auto', random_state=0,
# #                            subsample=1.0, tol=0.0001, validation_fraction=0.1,
# #                            verbose=0, warm_start=False)
# # 0.6666666666666667

# %%
# SMOTE (another version)
#
# smt = SMOTE()
# X_train, y_train = smt.fit_sample(X_train, y_train)
#
# X_train.shape
# y_train.shape
#
# type(X_train)
# pd.DataFrame(y_train)[0].value_counts()

# %%

ros = RandomOverSampler()
X_train, y_train = ros.fit_sample(X_train, y_train)

pd.DataFrame(y_train)[0].value_counts()

# %%

tuning_params = [{'n_estimators': [10, 50, 100, 200],
                  'class_weight': ['balanced_subsample', 'balanced']}]
rand_forest = RandomForestClassifier(random_state=99)

rand_forest = GridSearchCV(rand_forest, tuning_params,
                           scoring='f1', cv=5, n_jobs=3)

rand_forest.fit(X_train, y_train)

rand_forest_output = rand_forest.predict(pca_test_df)

rand_for_output_df = pd.Series(rand_forest_output)

print(rand_forest.best_estimator_)
print(rand_forest.score(X_test, y_test))

f1_score(rand_forest.predict(X_test), y_test)

# %%
# XG boost

base_model = xgb.XGBClassifier(random_state=99)

tuning_params = [{'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]
                  # ,'scale_pos_weight': [3, 5, 7, 9, 11, 13, 15]
                  }]

xgb_model = GridSearchCV(base_model,
                         tuning_params,
                         scoring='f1', cv=5, n_jobs=2)
xgb_model.fit(X_train, y_train)

print(xgb_model.best_estimator_)
print(xgb_model.score(X_test, y_test))

#
# %%
# Gradient boosting

tuning_params = [{'n_estimators': [200, 400, 600, 800], 'learning_rate': [0.25, 0.5, 0.75],
                  'max_depth': [3, 5, 7, 9], 'loss': ['exponential', 'deviance']
                  }]

base_model = GradientBoostingClassifier(random_state=0)

clf = GridSearchCV(base_model,
                   tuning_params,
                   scoring='f1', cv=5, n_jobs=2)

clf.fit(X_train, y_train)

print(clf.best_estimator_)
print(clf.score(X_test, y_test))

# %%


# pca_test_df.shape
test_output = clf.predict(pca_test_df)

test_output_df = pd.Series(test_output)

test_output_df.to_csv('/team-submission-GBDT.csv', index=False)
test_output_df.value_counts()
