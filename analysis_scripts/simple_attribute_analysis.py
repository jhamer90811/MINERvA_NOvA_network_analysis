"""
Author: Jesse Hamer
8/9/18

Analysis of simple attribute data obtained from MINERvA and NOvA networks.
"""

# Load data (Note: many regressors have already been pruned for collinearity and lackluster histograms.)

# Using only 10 of 88 extracted attributes.

import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from sklearn.feature_selection import f_classif, f_regression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Brief analysis of raw data

data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/MINERvA_NOvA_network_analysis/minerva-simple-00001-05000.csv', index_col = 0)
# Drop any rows with NaNs:
data = data.dropna()
data_means = data.mean()
data_stds = data.std()

# Note attributes with 0 std:

print(data_stds.where(data_stds == 0.0).dropna().to_string())
# Drop these, along with avg_concat_width and avg_split_width, which have essentially 0 variance
data = data.drop(columns=list(data_stds.where(data_stds == 0.0).dropna().index))

# Get histograms:
data_mins = data.min(0)
data_maxs = data.max(0)

for col in data.drop(columns=['genealogy','avg_concat_width', 'avg_split_width'])).columns:
    hist, bins = np.histogram(data[col], bins=50)
    plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='blue')
    x = np.linspace(data_mins.loc[col], data_maxs.loc[col], 10000)
    y = mlab.normpdf(x, data_means.loc[col], data_stds.loc[col])
    plt.plot(x, y, 'k--')
    plt.title('Histogram of {}'.format(col))
    # plt.axvline(data_means.loc[col], color='r', linewidth=1)
    plt.show()

# Get correlations:
data_corr = data.corr()
# Which variables are correlated at > 0.5?
print(data_corr.where(abs(data_corr) > 0.5).to_string())
# Drop any min_ or max_ prefaced attributes in favor of their avg_ counterparts. The avg_ attributes
# seem to have the most normal histograms, and the three tend to be highly correlated.

data = data.drop(columns = [c for c in data.columns if 'max' in c or 'min' in c])

# "num_conv_features" and "num_conv_features.1" were misnamed--these should be max_conv_features and
# min_conv_features, and accordingly will be dropped. Moreover, avg_conv_ker_area is actually avg_num_
# conv_features due to an error in the data collection code (see get_simple_attribues.py). Thus this
# will also be dropped.

data = data.drop(columns = ['num_conv_features', 'num_conv_features.1', 'avg_conv_ker_area'])

# This leaves 32 regressors altogether.

data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_all_acc_32_feat.csv')

#Normalize (to standard normal scale)

data_means = data.mean()
data_means.loc['final_accuracy', 'genealogy'] = 0
data_stds = data.std()
data_stds.loc['final_accuracy', 'genealogy'] = 1

data_normed = (data-data_means)/data_stds

data_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_all_acc_32_feat_normed.csv')


# Examine correlations.
data_corr = data.corr()
print(data_corr.where(abs(data_corr)>0.5).to_string())

# Drop all terms besides net_depth_avg, avg_IP_weights, prop_conv_into_pool, prop_horiz_kernels,
# avg_grid_reduction_width_total, avg_ratio_features_to_depth, avg_ratio_features_to_kerWidth,
# avg_ratio_features_to_kerHeight, avg_ratio_kerWidth_to_depth, avg_ratio_kerHeight_to_depth, genealogy,
# final_accuracy

# Load data w/ only 10 attributes

all_data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_all.csv', index_col=0)
data_corr = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_corr.csv', index_col=0)
norm_data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_std_normalized.csv', index_col=0)
# min_max_data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_min_max_normalized.csv', index_col=0)

# Get histograms:

data_means = all_data.mean()
data_means.loc['final_accuracy'] = 0
data_stds = all_data.std()
data_stds.loc['final_accuracy'] = 1
data_mins = all_data.min(0)
data_maxs = all_data.max(0)

# Plot histograms
for col in all_data.drop(columns=['genealogy']).columns:
    hist, bins = np.histogram(all_data[col], bins=50)
    plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='blue')
    x = np.linspace(data_mins.loc[col], data_maxs.loc[col], 10000)
    y = mlab.normpdf(x, data_means.loc[col], data_stds.loc[col])
    plt.plot(x, y, 'k--')
    plt.title('Histogram of {}'.format(col))
    # plt.axvline(data_means.loc[col], color='r', linewidth=1)
    plt.show()

# Linear regression models

y = all_data['final_accuracy']
X = all_data.drop(columns=['final_accuracy', 'genealogy'])
X_norm = norm_data.drop(columns=['final_accuracy', 'genealogy'])

lr = LinearRegression(True, False, True)
lr_norm = LinearRegression(True, True, True)

lr.fit(X, y)
lr_norm.fit(X_norm, y)

print('R^2 for non-normalized data: {}'.format(lr.score(X, y)))
print('R^2 for normalized data: {}'.format(lr_norm.score(X_norm, y)))

pca = PCA()
pca.fit(X_norm)
print(pca.explained_variance_ratio_)
X_norm_pca = pca.transform(X_norm)

regressors = list(X_norm.columns)
principal_components = [[(r, np.round(c,2)) for r, c in zip(regressors, pc)] for pc in pca.components_]

pc_names = ['PC_{}'.format(i) for i in range(1, 11)]
indices = list(norm_data.index)

norm_pca_data = pd.DataFrame(X_norm_pca, index=indices, columns=pc_names)
norm_pca_data = norm_pca_data.join(norm_data.loc[:,'genealogy':])

# plot top 2 principal components

x, y = norm_pca_data.loc[:, 'PC_1'], norm_pca_data.loc[:, 'PC_2']
acc = norm_pca_data.loc[:,'final_accuracy']


plt.scatter(x, y, 0.01, 'b')
plt.show()
plt.scatter(y,acc, 0.01, 'r')
plt.show()
plt.scatter(x, acc, 0.01, 'g')
plt.show()

# Redo this analysis, but remove all networks with accuracy < 0.1

high_acc_data = all_data.where(all_data['final_accuracy'] > 0.1).dropna()

high_acc_means = high_acc_data.mean()
high_acc_stds = high_acc_data.std()
high_acc_mins = high_acc_data.min(0)
high_acc_maxs = high_acc_data.max(0)

high_acc_normed = (high_acc_data - high_acc_means)/high_acc_stds

high_acc_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_high_acc.csv')
high_acc_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_high_acc_normed.csv')

y = high_acc_data['final_accuracy']
X = high_acc_data.drop(columns=['final_accuracy', 'genealogy'])
X_norm = high_acc_normed.drop(columns=['final_accuracy', 'genealogy'])

lr = LinearRegression(True, False, True)
lr_norm = LinearRegression(True, True, True)

lr.fit(X, y)
lr_norm.fit(X_norm, y)

print('R^2 for non-normalized data with acc > 0.1: {}'.format(lr.score(X, y)))
print('R^2 for normalized data with acc > 0.1: {}'.format(lr_norm.score(X_norm, y)))

pca = PCA()
pca.fit(X_norm)
print(pca.explained_variance_ratio_)
X_norm_pca = pca.transform(X_norm)

regressors = list(X_norm.columns)
high_acc_principal_components = [[(r, np.round(c, 2)) for r, c in zip(regressors, pc)] for pc in pca.components_]

pc_names = ['PC_{}'.format(i) for i in range(1, 11)]
indices = list(high_acc_normed.index)

norm_pca_data = pd.DataFrame(X_norm_pca, index=indices, columns=pc_names)
norm_pca_data = norm_pca_data.join(high_acc_normed.loc[:, 'genealogy':])

# %%

# General code with variable accuracy tolerance.

# How does model fit behave when we remove low-accuracy networks?

def get_high_acc_data(tol=0, save_data = False):
    """
    Truncates dataset below accuracy threshold tol and performs PCA and regression analysis. Saves datasets
    (including PCA-transformed data) to /pandas_dataframes directory.

    :param tol: accuracy threshold
    :param save_data: whether or not to save the dataframes.
    :return: None
    """
    all_data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_all.csv', index_col=0)

    high_acc_data = all_data.where(all_data['final_accuracy'] > tol).dropna()
    print('Shape of new data: {}'.format(high_acc_data.shape))

    high_acc_means = high_acc_data.mean()
    high_acc_stds = high_acc_data.std()

    high_acc_normed = (high_acc_data - high_acc_means)/high_acc_stds
    if save_data:
        high_acc_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_{}.csv'.format(tol))
        high_acc_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_{}_normed.csv'.format(tol))

    y = high_acc_data['final_accuracy']
    X = high_acc_data.drop(columns=['final_accuracy', 'genealogy'])
    X_norm = high_acc_normed.drop(columns=['final_accuracy', 'genealogy'])

    lr = LinearRegression(True, False, True)
    lr_norm = LinearRegression(True, True, True)

    lr.fit(X, y)
    lr_norm.fit(X_norm, y)

    print('R^2 for non-normalized data with acc > {}: {}'.format(tol, lr.score(X, y)))

    pca = PCA()
    pca.fit(X_norm)
    print(pca.explained_variance_ratio_)
    X_norm_pca = pca.transform(X_norm)

    regressors = list(X_norm.columns)
    high_acc_principal_components = [[(r, np.round(c, 2)) for r, c in zip(regressors, pc)] for pc in pca.components_]

    pc_names = ['PC_{}'.format(i) for i in range(1, 11)]
    indices = list(high_acc_normed.index)

    norm_pca_data = pd.DataFrame(X_norm_pca, index=indices, columns=pc_names)
    norm_pca_data = norm_pca_data.join(high_acc_normed.loc[:, 'genealogy':])
    if save_data:
        norm_pca_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_{}_pca.csv'.format(tol))


for tol in np.linspace(0.0, 0.1, 21):

    get_high_acc_data(tol)

# %%
# Best accuracy cutoff is 0.054

tol = 0.054
all_data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_all.csv', index_col=0)

high_acc_data = all_data.where(all_data['final_accuracy'] > tol).dropna()
print('Shape of new data: {}'.format(high_acc_data.shape))

high_acc_means = high_acc_data.mean()
high_acc_means.loc['final_accuracy'] = 0
high_acc_stds = high_acc_data.std()
high_acc_stds.loc['final_accuracy'] = 1
high_acc_mins = high_acc_data.min(0)
high_acc_maxs = high_acc_data.max(0)

high_acc_normed = (high_acc_data - high_acc_means)/high_acc_stds

high_acc_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_{}.csv'.format(tol))
high_acc_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_{}_normed.csv'.format(tol))

y = high_acc_data['final_accuracy']
X = high_acc_data.drop(columns=['final_accuracy', 'genealogy'])
X_norm = high_acc_normed.drop(columns=['final_accuracy', 'genealogy'])

lr = LinearRegression(True, False, True)
lr_norm = LinearRegression(True, True, True)

lr.fit(X, y)
lr_norm.fit(X_norm, y)

print('R^2 for non-normalized data with acc > {}: {}'.format(tol, lr.score(X, y)))

pca = PCA()
pca.fit(X_norm)
print(pca.explained_variance_ratio_)
X_norm_pca = pca.transform(X_norm)

regressors = list(X_norm.columns)
high_acc_principal_components = [[(r, np.round(c, 2)) for r, c in zip(regressors, pc)] for pc in pca.components_]

pc_names = ['PC_{}'.format(i) for i in range(1, 11)]
indices = list(high_acc_normed.index)

norm_pca_data = pd.DataFrame(X_norm_pca, index=indices, columns=pc_names)
norm_pca_data = norm_pca_data.join(high_acc_normed.loc[:, 'genealogy':])

norm_pca_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_{}_pca.csv'.format(tol))

# Plot histograms of attributes for accuracy > 0.054

for col in high_acc_data.drop(columns=['genealogy']).columns:
    hist, bins = np.histogram(high_acc_data[col], bins=50)
    plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='blue')
    x = np.linspace(high_acc_mins.loc[col], high_acc_maxs.loc[col], 10000)
    y = mlab.normpdf(x, high_acc_means.loc[col], high_acc_stds.loc[col])
    plt.plot(x, y, 'k--')
    plt.title('Histogram of {} for Accuracy > 0.054'.format(col))
    # plt.axvline(data_means.loc[col], color='r', linewidth=1)
    plt.show()

# %%

# Looking at most parsimonious quadratic model. (R^2 = 0.3862)
# This model allows all 32 attributes obtained in the introductory analysis. As such, there will be
# considerable multicollinearity.

lin_terms = ['avg_ratio_features_to_kerWidth','avg_grid_reduction_width_total','avg_grid_reduction_height_total','avg_ratio_kerWidth_to_depth','avg_num_conv_features','avg_grid_reduction_width_consecutive','num_relu','avg_ratio_features_to_depth','prop_vert_kernels','avg_ratio_kerArea_to_depth','avg_ratio_features_to_kerHeight','prop_1x1_conv','num_pooling_layers']
int_terms = ['avg_grid_reduction_width_total:avg_ratio_kerWidth_to_depth','avg_ratio_kerWidth_to_depth:avg_grid_reduction_width_consecutive','num_relu:avg_ratio_features_to_kerHeight','avg_ratio_features_to_kerWidth:num_relu','avg_stride_h:avg_ratio_kerHeight_to_depth','avg_ratio_features_to_kerWidth:avg_ratio_features_to_kerHeight','avg_grid_reduction_width_consecutive:prop_vert_kernels','avg_ratio_kerWidth_to_depth:num_conv_layers','prop_1x1_conv:prop_square_kernels','num_conv_layers:avg_stride_w','avg_ratio_features_to_kerWidth:prop_vert_kernels','net_depth_avg:avg_IP_neurons','avg_ratio_features_to_kerWidth:avg_ratio_features_to_kerArea','num_pooling_layers:avg_IP_neurons','num_conv_layers:avg_IP_neurons','avg_grid_reduction_width_consecutive:avg_ratio_kerArea_to_depth','avg_grid_reduction_width_consecutive:avg_grid_reduction_area_consecutive','avg_ratio_features_to_depth:avg_stride_w','avg_ratio_features_to_kerWidth:avg_ratio_kerWidth_to_depth','avg_ratio_features_to_kerWidth:avg_grid_reduction_width_total','avg_ratio_features_to_kerWidth:avg_stride_h','avg_num_conv_features:num_relu','prop_square_kernels:avg_ratio_features_to_kerArea','avg_num_conv_features:avg_stride_w','num_relu:avg_IP_neurons','avg_num_conv_features:avg_grid_reduction_width_consecutive','avg_ratio_features_to_depth:num_sigmoid','avg_ratio_kerArea_to_depth:num_conv_layers','num_relu:avg_ratio_features_to_kerArea','num_conv_layers:prop_conv_into_pool','avg_ratio_features_to_kerHeight:avg_grid_reduction_area_consecutive','avg_grid_reduction_area_consecutive:avg_grid_reduction_height_consecutive','avg_stride_w:avg_IP_neurons','avg_grid_reduction_width_total:avg_grid_reduction_width_consecutive','avg_ratio_features_to_kerHeight:avg_grid_reduction_height_consecutive','num_relu:prop_conv_into_pool','avg_ratio_kerWidth_to_depth:avg_IP_neurons','avg_ratio_features_to_kerHeight:prop_pool_into_pool','prop_vert_kernels:prop_horiz_kernels','avg_grid_reduction_area_consecutive:prop_horiz_kernels','num_relu:avg_stride_h']
int_terms = [(t[:t.find(':')], t[t.find(':')+1:]) for t in int_terms]
data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_simple_0.054_all_features_normed.csv', index_col=0)
data_quad = data.loc[:, ['final_accuracy'] + lin_terms]

for t in int_terms:
    s = data[t[0]]*data[t[1]]
    s.name = '{}:{}'.format(t[0], t[1])
    data_quad = data_quad.join(s)

lr = LinearRegression()

data_quad_X = data_quad.iloc[:, 1:]
data_quad_y = np.log(data_quad['final_accuracy']+1)
data_quad_y.name = 'log(final_accuracy + 1)'
lr.fit(data_quad_X, data_quad_y)

lr.score()

num_bins = 10

bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)

bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]

labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]

data_quad_labels = pd.Series(labels, name='accuracy_labels')

f_regression(data_quad_X, data_quad_y)

kmeans = KMeans(n_clusters=10)

kmeans.fit(data_quad_X, data_quad_labels)

meanshift = MeanShift()
meanshift.fit(data_quad_X, data_quad_y)

kmeans_labels_pred = kmeans.predict(data_quad_X)

meanshift_labels_pred = meanshift.predict(data_quad_X)

print('Adjusted rand score for K-means: {}.'.format(metrics.adjusted_rand_score(labels, kmeans_labels_pred)))
print('Adjusted rand score for Mean Shift: {}'. format(metrics.adjusted_rand_score(labels, meanshift_labels_pred)))

print('Average Inertia for K-means: {}'.format(kmeans.inertia_/data_quad_X.shape[0]))

print('Mean Shift predicts {} labels.'.format(max(meanshift.labels_)+1))

# Try k-means for various values of k:

pca = PCA(10)
data_quad_pca_X = pca.fit_transform(data_quad_X)

for k in range(2, 50, 2):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_quad_pca_X)
    labels_pred = kmeans.predict(data_quad_pca_X)

    bins = np.linspace(min_acc, max_acc + 1e-8, k + 1)

    bins = [(i, bins[i], bins[i + 1]) for i in range(k)]

    labels = [[i for i in range(k) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]

    print('k = {}; Adj. Rand = {}; Inertia = {}; Adj. Mutual Info = {}; Homogeneity = {}; Completeness = {}; Silhouette = {}; CH = {}'.format(
        k,
        metrics.adjusted_rand_score(labels, labels_pred),
        kmeans.inertia_,
        metrics.adjusted_mutual_info_score(labels, labels_pred),
        metrics.homogeneity_score(labels, labels_pred),
        metrics.completeness_score(labels, labels_pred),
        metrics.silhouette_score(data_quad_pca_X, labels_pred, sample_size=100),
        metrics.calinski_harabaz_score(data_quad_pca_X, labels_pred)
    ))

meanshift = MeanShift()
meanshift.fit(data_quad_pca_X, data_quad_y)
meanshift_labels_pred = meanshift.predict(data_quad_pca_X)

print('Adjusted rand score for Mean Shift: {}'. format(metrics.adjusted_rand_score(labels, meanshift_labels_pred)))

print('Mean Shift predicts {} labels.'.format(max(meanshift.labels_)+1))

print('Mean Shift Silhouette: {}; Mean Shift Calinski-Harabaz: {}'.format(metrics.silhouette_score(data_quad_pca_X, meanshift_labels_pred, sample_size=100),
                                                                          metrics.calinski_harabaz_score(data_quad_pca_X, meanshift_labels_pred)))
num_bins = 333

bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)

bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]

labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]

print('Adjusted rand score for Mean Shift: {}'. format(metrics.adjusted_rand_score(labels, meanshift_labels_pred)))

# Redo Mean-shift with top 3 principal components:

pca = PCA(3)
data_quad_pca_X = pca.fit_transform(data_quad_X)

meanshift = MeanShift()
meanshift.fit(data_quad_pca_X, data_quad_y)
meanshift_labels_pred = meanshift.predict(data_quad_pca_X)

num_bins = max(meanshift.labels_)

bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)

bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]

labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]

print('Adjusted rand score for Mean Shift: {}'. format(metrics.adjusted_rand_score(labels, meanshift_labels_pred)))

print('Adjusted rand score for Mean Shift: {}'. format(metrics.adjusted_rand_score(labels, meanshift_labels_pred)))

print('Mean Shift predicts {} labels.'.format(max(meanshift.labels_)+1))

print('Mean Shift Silhouette: {}; Mean Shift Calinski-Harabaz: {}'.format(metrics.silhouette_score(data_quad_pca_X, meanshift_labels_pred, sample_size=100),
                                                                          metrics.calinski_harabaz_score(data_quad_pca_X, meanshift_labels_pred)))
# Plot clusters and raw PCA data.
x = data_quad_pca_X[:, 0]
y = data_quad_pca_X[:, 1]
z = data_quad_pca_X[:, 2]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xbound(min(x), max(x))
ax.set_ybound(min(y), max(y))
ax.set_zbound(min(z), max(z))

ax.scatter(x, y, z, c='b')
plt.show()

pca = PCA(2)
data_quad_pca_X = pca.fit_transform(data_quad_X)

kmeans = KMeans(3)
kmeans.fit(data_quad_pca_X)

num_bins = 3
bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)
bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]
labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]
labels_pred = kmeans.predict(data_quad_pca_X)

pts0_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 0]
pts1_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 1]
pts2_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 2]

pts0_pred = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels_pred[i] == 0]
pts1_pred = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels_pred[i] == 1]
pts2_pred = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels_pred[i] == 2]

x0t, y0t = [p[0] for p in pts0_true], [p[1] for p in pts0_true]
x1t, y1t = [p[0] for p in pts1_true], [p[1] for p in pts1_true]
x2t, y2t = [p[0] for p in pts2_true], [p[1] for p in pts2_true]

ax = plt.axes()

ax.set_xbound(min(x0t + x1t + x2t), max(x0t + x1t + x2t))
ax.set_ybound(min(y0t + y1t + y2t), max(y0t + y1t + y2t))

ax.scatter(x0t, y0t, c='b')
ax.scatter(x1t, y1t, c='r')
ax.scatter(x2t, y2t, c='g')
plt.show()

x0p, y0p = [p[0] for p in pts0_pred], [p[1] for p in pts0_pred]
x1p, y1p = [p[0] for p in pts1_pred], [p[1] for p in pts1_pred]
x2p, y2p = [p[0] for p in pts2_pred], [p[1] for p in pts2_pred]

ax = plt.axes()

ax.set_xbound(min(x0p + x1p + x2p), max(x0p + x1p + x2p))
ax.set_ybound(min(y0p + y1p + y2p), max(y0p + y1p + y2p))

ax.scatter(x0p, y0p, c='b')
ax.scatter(x1p, y1p, c='r')
ax.scatter(x2p, y2p, c='g')
plt.show()

pca = PCA(3)
data_quad_pca_X = pca.fit_transform(data_quad_X)

pts0_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 0]
pts1_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 1]
pts2_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 2]

pts0_pred = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels_pred[i] == 0]
pts1_pred = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels_pred[i] == 1]
pts2_pred = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels_pred[i] == 2]

x0t, y0t, z0t = [p[0] for p in pts0_true], [p[1] for p in pts0_true], [p[2] for p in pts0_true]
x1t, y1t, z1t = [p[0] for p in pts1_true], [p[1] for p in pts1_true], [p[2] for p in pts1_true]
x2t, y2t, z2t = [p[0] for p in pts2_true], [p[1] for p in pts2_true], [p[2] for p in pts2_true]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xbound(min(x0t + x1t + x2t), max(x0t + x1t + x2t))
ax.set_ybound(min(y0t + y1t + y2t), max(y0t + y1t + y2t))
ax.set_zbound(min(z0t + z1t + z2t), max(z0t + z1t + z2t))

ax.scatter(x0t, y0t, z0t, s=1, c='b', alpha=0.1)
ax.scatter(x1t, y1t, z1t, s=10, c='r', alpha=0.2)
ax.scatter(x2t, y2t, z2t, s=50, c='g', alpha=1)
ax.view_init(45, -15)
ax.text2D(0.5, 0.95, 'True Label Accuracy Clustering', transform=ax.transAxes)
plt.show()

x0p, y0p, z0p = [p[0] for p in pts0_pred], [p[1] for p in pts0_pred], [p[2] for p in pts0_pred]
x1p, y1p, z1p = [p[0] for p in pts1_pred], [p[1] for p in pts1_pred], [p[2] for p in pts1_pred]
x2p, y2p, z2p = [p[0] for p in pts2_pred], [p[1] for p in pts2_pred], [p[2] for p in pts2_pred]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xbound(min(x0p + x1p + x2p), max(x0p + x1p + x2p))
ax.set_ybound(min(y0p + y1p + y2p), max(y0p + y1p + y2p))
ax.set_zbound(min(z0p + z1p + z2p), max(z0p + z1p + z2p))

ax.scatter(x0p, y0p, z0p, c='b')
ax.scatter(x1p, y1p, z1p, c='r')
ax.scatter(x2p, y2p, z2p, c='g')
ax.text2D(0.5, 0.95, 'K-Means Label Accuracy Clustering', transform=ax.transAxes)
plt.show()

# 2D heatmap for first two principal components:

pca = PCA(2)
data_quad_pca_X = pca.fit_transform(data_quad_X)

x = data_quad_pca_X[:, 0]
y = data_quad_pca_X[:, 1]
z = data_quad_y+1

f = plt.figure()
ax = plt.axes()

ax.set_xbound(min(x), max(x))
ax.set_ybound(min(y), max(y))

s = ax.scatter(x, y, c=z, s=5, edgecolor='', cmap=plt.get_cmap('YlOrRd'))

ax.set_title('Top 2 Principal Components Accuracy Heatmap')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

f.colorbar(s)

plt.show()

# Try binning first
pca = PCA(2)
data_quad_pca_X = pca.fit_transform(data_quad_X)

num_bins = 4
bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)
bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]
labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]
labels_pred = kmeans.predict(data_quad_pca_X)

x = data_quad_pca_X[:, 0]
y = data_quad_pca_X[:, 1]
z = labels

f = plt.figure()
ax = plt.axes()

ax.set_xbound(min(x), max(x))
ax.set_ybound(min(y), max(y))

s = ax.scatter(x, y, c=z, s=50, edgecolor='')

ax.set_title('Top 2 Principal Components Accuracy Heatmap')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

f.colorbar(s)

plt.show()

# 3d heatmap

pca = PCA(3)
data_quad_pca_X = pca.fit_transform(data_quad_X)

x = data_quad_pca_X[:, 0]
y = data_quad_pca_X[:, 1]
z = data_quad_pca_X[:, 2]
w = data_quad_y

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xbound(min(x), max(x))
ax.set_ybound(min(y), max(y))
ax.set_zbound(min(z), max(z))

s = ax.scatter(x, y, z, s=10, c=w, edgecolor='', cmap=plt.get_cmap('YlOrRd'))
ax.text2D(0.5, 0.95, 'First 3 Principal Components Accuracy Heatmap', transform=ax.transAxes)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.view_init(azim=15)

f.colorbar(s)

plt.show()

# Color according to top 4 bins

pca = PCA(2)
data_quad_pca_X = pca.fit_transform(data_quad_X)

num_bins = 4
bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)
bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]
labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]
labels_pred = kmeans.predict(data_quad_pca_X)

pts0_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 0]
pts1_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 1]
pts2_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 2]
pts3_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 3]

x0t, y0t = [p[0] for p in pts0_true], [p[1] for p in pts0_true]
x1t, y1t = [p[0] for p in pts1_true], [p[1] for p in pts1_true]
x2t, y2t = [p[0] for p in pts2_true], [p[1] for p in pts2_true]
x3t, y3t = [p[0] for p in pts3_true], [p[1] for p in pts3_true]

ax = plt.axes()

ax.set_xbound(min(x0t + x1t + x2t + x3t), max(x0t + x1t + x2t + x3t))
ax.set_ybound(min(y0t + y1t + y2t + y3t), max(y0t + y1t + y2t + y3t))

ax.scatter(x0t, y0t, c='b', alpha=0.2)
ax.scatter(x1t, y1t, c='r', alpha=0.2)
ax.scatter(x2t, y2t, c='g', alpha=0.2)
ax.scatter(x3t, y3t, c='y', alpha=0.2)
ax.set_title('First Two Principal Components. b=0, r = 1, g = 2, 3 = y')
plt.show()

# Try with 3 dimensions
pca = PCA(3)
data_quad_pca_X = pca.fit_transform(data_quad_X)

pts0_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 0]
pts1_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 1]
pts2_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 2]
pts3_true = [data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == 3]

x0t, y0t, z0t = [p[0] for p in pts0_true], [p[1] for p in pts0_true], [p[2] for p in pts0_true]
x1t, y1t, z1t = [p[0] for p in pts1_true], [p[1] for p in pts1_true], [p[2] for p in pts1_true]
x2t, y2t, z2t = [p[0] for p in pts2_true], [p[1] for p in pts2_true], [p[2] for p in pts2_true]
x3t, y3t, z3t = [p[0] for p in pts3_true], [p[1] for p in pts3_true], [p[2] for p in pts3_true]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xlim(min(x0t + x1t + x2t + x3t), max(x0t + x1t + x2t + x3t))
ax.set_ylim(min(y0t + y1t + y2t + y3t), max(y0t + y1t + y2t + y3t))
ax.set_zlim(min(z0t + z1t + z2t + z3t), max(z0t + z1t + z2t + z3t))

ax.scatter(x0t, y0t, z0t, s=10, c='b', alpha=0.2)
ax.scatter(x1t, y1t, z1t, s=10, c='r', alpha=0.2)
ax.scatter(x2t, y2t, z2t, s=10, c='g', alpha=0.2)
ax.scatter(x3t, y3t, z3t, s=10, c='y', alpha=0.2)

ax.text2D(0.5, 0.95, 'True Label Accuracy Clustering', transform=ax.transAxes)
plt.show()

# Show growth of bins as accuracy is allowed to be lower.

pca = PCA(3)
data_quad_pca_X = pca.fit_transform(data_quad_X)

num_bins = 6
min_acc, max_acc = min(data_quad_y), max(data_quad_y)
bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)
bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]
labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]

pts = [[data_quad_pca_X[i, :2] for i in range(len(labels)) if labels[i] == k] for k in range(num_bins)]
xmin, xmax = min(data_quad_pca_X[:,0]), max(data_quad_pca_X[:,0])
ymin, ymax = min(data_quad_pca_X[:,1]), max(data_quad_pca_X[:,1])

for i in reversed(range(num_bins)):

    current_pts = pts[i:]
    acc = bins[i][1]
    num_points = 0
    f = plt.figure(figsize=(10,6))
    ax = plt.axes()
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for pt in reversed(current_pts):
        x, y = [p[0] for p in pt], [p[1] for p in pt]
        ax.scatter(x, y, edgecolor='', s=10)
        num_points += len(pt)

    ax.set_title('First 2 Principal Components; {} <= log(accuracy+1) <= {}; Number of Networks = {}'.format(np.round(acc, 4), np.round(max_acc, 4), num_points))
    plt.show()

pca = PCA(3)
data_quad_pca_X = pca.fit_transform(data_quad_X)

num_bins = 6
bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)
bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]
labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in data_quad_y]

pts = [[data_quad_pca_X[i, :] for i in range(len(labels)) if labels[i] == k] for k in range(num_bins)]
xmin, xmax = min(data_quad_pca_X[:,0]), max(data_quad_pca_X[:,0])
ymin, ymax = min(data_quad_pca_X[:,1]), max(data_quad_pca_X[:,1])
zmin, zmax = min(data_quad_pca_X[:,2]), max(data_quad_pca_X[:,2])

for i in reversed(range(num_bins)):

    current_pts = pts[i:]
    acc = bins[i][1]
    num_points = 0
    f = plt.figure(figsize=(8,6))
    ax = mplot3d.axes3d.Axes3D(f)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    for pt in reversed(current_pts):
        x, y, z = [p[0] for p in pt], [p[1] for p in pt], [p[2] for p in pt]
        ax.scatter(x, y, z, edgecolor='', s=10)
        num_points += len(pt)

    ax.text2D(0.05, 0.95, 'First 3 Principal Components; {} <= log(accuracy+1) <= {}; Number of Networks = {}'.format(np.round(acc, 4), np.round(max_acc, 4), num_points), transform=ax.transAxes)
    plt.show()

centroid = data_quad_pca_X.mean(axis=0)
distances = np.array([np.linalg.norm(data_quad_pca_X[i,:]-centroid) for i in range(data_quad_pca_X.shape[0])])

ax = plt.axes()
ax.scatter(distances, data_quad_y)
ax.set_xlabel('Distance from Centroid')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

ax = plt.axes()
ax.scatter(distances**2, data_quad_y)
ax.set_xlabel('Distance^2 from Centroid')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Try to determine a relationship between accuracy and distance from nearest cluster center.
kmeans = KMeans(2)
kmeans.fit(data_quad_pca_X)

clusters = [np.array(c) for c in kmeans.cluster_centers_]
cluster_dist = [min([np.linalg.norm(p-c) for c in clusters]) for p in [data_quad_pca_X[i, :] for i in range(data_quad_pca_X.shape[0])]]
cluster_dist = np.array(cluster_dist)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Fit a line

line_coef = np.polyfit(cluster_dist, data_quad_y, 1)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.scatter(cluster_dist, line_coef[0]*cluster_dist + line_coef[1], linewidth=1)
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Fit a parabola

parabola_coef = np.polyfit(cluster_dist, data_quad_y, 2)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.scatter(cluster_dist, parabola_coef[0]*cluster_dist**2 + parabola_coef[1]*cluster_dist + parabola_coef[2], linewidth=1)
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Fit a hyperbola

hyperbola_coef = np.polyfit(1/np.sqrt(cluster_dist), data_quad_y, 1)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.scatter(cluster_dist, hyperbola_coef[0]*(1/np.sqrt(cluster_dist)) + hyperbola_coef[1], linewidth=1)
ax.set_title('Graph of y = m/sqrt(distance from cluster) + b')
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

lr = LinearRegression()

lr.fit(1/np.sqrt(cluster_dist.reshape(-1,1)), data_quad_y)

# Where are the clusters?

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(zmin, zmax)

for pt in clusters:
    x, y, z = [pt[0]], [pt[1]], [pt[2]]
    ax.scatter(x, y, z, edgecolor='', s=100, c='k')

ax.text2D(0.05, 0.95, 'Position of Cluster Centers', transform=ax.transAxes)
plt.show()

for i in reversed(range(num_bins)):

    current_pts = pts[i:]
    acc = bins[i][1]
    num_points = 0
    f = plt.figure(figsize=(8, 6))
    ax = mplot3d.axes3d.Axes3D(f)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    for pt in clusters:
        x, y, z = [pt[0]], [pt[1]], [pt[2]]
        ax.scatter(x, y, z, edgecolor='', s=100, c='k')

    for pt in reversed(current_pts):
        x, y, z = [p[0] for p in pt], [p[1] for p in pt], [p[2] for p in pt]
        ax.scatter(x, y, z, edgecolor='', s=10)
        num_points += len(pt)

    ax.text2D(0.05, 0.95, 'First 3 Principal Components; {} <= log(accuracy+1) <= {}; Number of Networks = {}'.format(np.round(acc, 4), np.round(max_acc, 4), num_points), transform=ax.transAxes)
    plt.show()

# Redo, but only measure distance from the first (largest) cluster.
cluster_dist = [np.linalg.norm(p-clusters[0]) for p in [data_quad_pca_X[i, :] for i in range(data_quad_pca_X.shape[0])]]
cluster_dist = np.array(cluster_dist)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Fit a line

line_coef = np.polyfit(cluster_dist, data_quad_y, 1)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.scatter(cluster_dist, line_coef[0]*cluster_dist + line_coef[1], linewidth=1)
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Fit a parabola

parabola_coef = np.polyfit(cluster_dist, data_quad_y, 2)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.scatter(cluster_dist, parabola_coef[0]*cluster_dist**2 + parabola_coef[1]*cluster_dist + parabola_coef[2], linewidth=1)
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# Fit a hyperbola

hyperbola_coef = np.polyfit(1/np.sqrt(cluster_dist), data_quad_y, 1)

ax = plt.axes()
ax.scatter(cluster_dist, data_quad_y)
ax.scatter(cluster_dist, hyperbola_coef[0]*(1/np.sqrt(cluster_dist)) + hyperbola_coef[1], linewidth=1)
ax.set_title('Graph of y = m/sqrt(distance from cluster) + b')
ax.set_xlabel('Distance from Densest Cluster Center')
ax.set_ylabel('log(accuracy + 1)')
plt.show()

# What are the dispersions of accuracy bins about the first cluster center?
stds = [np.std([np.linalg.norm(p-clusters[0]) for p in pt]) for pt in pts]

for i, s in enumerate(stds):
    print('Dispersion for bin {} is {}.'.format(i, s))



