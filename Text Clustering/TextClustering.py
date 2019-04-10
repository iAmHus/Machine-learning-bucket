# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# %%

input_text = pd.read_csv('datasets/team/input.dat', header=None)
input_text.columns = ['sentence_sparse']
input_text = input_text.apply(lambda x: x['sentence_sparse'].split(" "), axis=1)

feature_map = pd.read_csv('datasets/team/features.dat', header=None, squeeze=True).to_dict()


def create_sentence(row_val, feature_map):
    map_values = {}
    len_of_data = (len(row_val) - 1)

    for i in range(0, len_of_data, 2):
        map_values[row_val[i]] = row_val[i + 1]

    # map_values
    # Convert map into a sentence
    sentence = []
    for key, value in map_values.items():
        multiple_words = ((' ' + feature_map[int(key)]) * int(value))
        sentence.append(multiple_words.lstrip())

    return ' '.join(sentence)


type(input_text)

input_text = input_text.apply(lambda row: create_sentence(row, feature_map))

input_text.reset_index(drop=True, inplace=True)

input_text[1]

# %%
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(input_text)

svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
svd.fit(X)
# print(sum(svd.explained_variance_ratio_))
X = pd.DataFrame(svd.transform(X))
print(X.shape)



# %%

def fit(input, k, tolerance, max_iter):
    centroids = random_centroids(input, k)

    for iteration in range(max_iter):
        clusters = {}

        initialize_clusters(clusters, k)

        # Distance b/w the point & cluster; to choose -> nearest centroid

        for features in input:
            distances = get_distances(centroids, features)

            clusters[distances.index(min(distances))].append(features)

        prev_centroids = dict(centroids)

        # Avg the cluster data points to re-calculate the centroids
        centroids = recalc_centroids(centroids, clusters)

        stop_flag = False

        for c in centroids:
            if step_diff(centroids[c], prev_centroids[c]) > tolerance:
                stop_flag = True

        # Stop if the centroids don't change much (change < tolerance defined)
        if not stop_flag:
            return clusters, centroids


def recalc_centroids(centroids, clusters):
    for cluster in clusters:
        centroids[cluster] = np.average(clusters[cluster], axis=0)

    return centroids


def get_distances(centroids, features):
    distances = []
    for each_centroid in centroids:
        distances.append(cosine_similarity(features.reshape(-1, 500),
                                           centroids[each_centroid].reshape(-1, 500)))
    return distances


def initialize_clusters(clusters, k):
    for k_val in range(k):
        clusters[k_val] = []


def predict(data, centroids):
    distances = []
    for centroid in centroids:
        # distances.append(l2_norm(data - centroids[centroid]))
        distances.append(cosine_similarity(data.values.reshape(-1, 500),
                                           centroids[centroid].reshape(-1, 500)))

    # print(distances)
    best_cluster = distances.index(min(distances))
    return best_cluster


# Difference in percentage - in original and current centroid !
def step_diff(current_centroid, original_centroid):
    return np.sum((current_centroid - original_centroid) / original_centroid * 100.0)


def random_centroids(input, k):
    centroids = {}
    for i in range(0, k, 1):
        centroids[i] = input[i]
    return centroids


# %%

# clf = K_Means()
# def fit(input_data, no_of_components=7, tolerance=0.001, max_iter=300):

clusters, centroids = fit(X.values, 7, 0.001, 500)
style.use('ggplot')

colors = ["no_of_components", "m", "y", "r", "g", "c", "b"] * 10

for centroid in centroids:
    plt.scatter(centroids[centroid][0], centroids[centroid][1], marker="o", s=150)

for classification in clusters:
    color = colors[classification]
    for feature_set in clusters[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker="x", color=color, s=150, linewidths=5)

plt.show()

# %%
# clusters, centroids = fit(X.values)
clusters, centroids = fit(X.values, 7, 0.001, 300)

#%%
output_list = []
for each_val in X.iterrows():
    output_list.append(predict(each_val[1], centroids) + 1)

output_series = pd.Series(output_list)

print(output_series.value_counts())

output_series.to_csv('datasets/team/output/output.csv', index=False)



# %%
from sklearn.metrics import silhouette_score

print(silhouette_score(X, output_series, metric='cosine'))

