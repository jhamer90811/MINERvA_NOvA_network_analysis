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
import os
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, Perceptron
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from sklearn.manifold import Isomap, TSNE
from sklearn.feature_selection import f_classif, f_regression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Load data

data = pd.DataFrame()

head_dir = '/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/output_dir/complex_attributes'

for file in os.listdir(head_dir):
    data = data.append(pd.read_csv(head_dir + '/' + file, index_col=0))

data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_raw.csv')

data = data.drop(columns=['initial_accuracy'] + [c for c in data.columns if 'betti_0' in c])

data_R2 = data.iloc[:, -5:]
data_R2.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_R2.csv')

data = data.iloc[:, :-5]
data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_all.csv')

# Average Consecutive Bottleneck
data_acb = data.loc[:, ['final_accuracy'] + [f for f in data.columns if '_acb' in f]].dropna()
# First-Last Bottleneck
data_flb = data.loc[:, ['final_accuracy'] + [f for f in data.columns if '_flb' in f and '_flb2' not in f]].dropna()
# First Layer Bottleneck
data_flb2 = data.loc[:, ['final_accuracy'] + [f for f in data.columns if '_flb2' in f]].dropna()
# First Layer Minimal Alpha^2
data_flma = data.loc[:, ['final_accuracy'] + [f for f in data.columns if '_flma' in f]].dropna()
# Total Bottleneck Variation
data_tbv = data.loc[:, ['final_accuracy'] + [f for f in data.columns if '_tbv' in f]].dropna()

data_acb.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_acb.csv')
data_flb.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_flb.csv')
data_flb2.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_flb2.csv')
data_flma.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_flma.csv')
data_tbv.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_tbv.csv')

# Normalize because scoring functions vary in scale

mean_all = data.mean()
std_all = data.std()
mean_all['final_accuracy'] = 0
std_all['final_accuracy'] = 1
normed_all = (data-mean_all)/std_all
normed_all.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_all_normed.csv')

mean_acb = data_acb.mean()
std_acb = data_acb.std()
mean_acb['final_accuracy'] = 0
std_acb['final_accuracy'] = 1
normed_acb = (data_acb-mean_acb)/std_acb
normed_acb.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_acb_normed.csv')

mean_flb = data_flb.mean()
std_flb = data_flb.std()
mean_flb['final_accuracy'] = 0
std_flb['final_accuracy'] = 1
normed_flb = (data_flb-mean_flb)/std_flb
normed_flb.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_flb_normed.csv')

mean_flb2 = data_flb2.mean()
std_flb2 = data_flb2.std()
mean_flb2['final_accuracy'] = 0
std_flb2['final_accuracy'] = 1
normed_flb2 = (data_flb2-mean_flb2)/std_flb2
normed_flb2.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_flb2_normed.csv')

mean_flma = data_flma.mean()
std_flma = data_flma.std()
mean_flma['final_accuracy'] = 0
std_flma['final_accuracy'] = 1
normed_flma = (data_flma-mean_flma)/std_flma
normed_flma.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_flma_normed.csv')

mean_tbv = data_tbv.mean()
std_tbv = data_tbv.std()
mean_tbv['final_accuracy'] = 0
std_tbv['final_accuracy'] = 1
normed_tbv = (data_tbv-mean_tbv)/std_tbv
normed_tbv.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_tbv_normed.csv')

# Get high-accuracy datasets w/ all attributes

tol = 0.055
normed_all.where(normed_all['final_accuracy']>tol).dropna().to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_all_normed.csv')
normed_acb.where(normed_acb['final_accuracy']>tol).dropna().to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_acb_normed.csv')
normed_flb.where(normed_flb['final_accuracy']>tol).dropna().to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_flb_normed.csv')
normed_flb2.where(normed_flb2['final_accuracy']>tol).dropna().to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_flb2_normed.csv')
normed_flma.where(normed_flma['final_accuracy']>tol).dropna().to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_flma_normed.csv')
normed_tbv.where(normed_tbv['final_accuracy']>tol).dropna().to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_0.055_tbv_normed.csv')

data_mins = data.min()
data_maxs = data.max()
# Histograms of complex attributes.
for col in data.columns[1:]:
    hist, bins = np.histogram(data[col], bins=50)
    plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), color='blue')
    x = np.linspace(data_mins.loc[col], data_maxs.loc[col], 10000)
    y = mlab.normpdf(x, mean_all.loc[col], std_all.loc[col])
    plt.plot(x, y, 'k--')
    plt.title('Histogram of {}'.format(col))
    # plt.axvline(data_means.loc[col], color='r', linewidth=1)
    plt.show()

# Correlations
all_corr = normed_all.corr()

acb_corr = normed_acb.corr()

flb_corr = normed_flb.corr()

flb2_corr = normed_flb2.corr()

flma_corr = normed_flma.corr()

tbv_corr = normed_tbv.corr()

# Drop poorest performing regressors
drop_cols = [c for c in data.columns if '_acb' in c or ('_flb' in c and '_flb2' not in c)]
drop_cols = drop_cols + ['nonzero_activations_flb2', 'horiz_spread_flb2', 'horiz_sd_flb2', 'min_alpha2_flb2',
                         'num_persistent_holes_flb2', 'num_delaunay_edges_flb2', 'vert_spread_flb2']
drop_cols = drop_cols + ['horiz_spread_flma', 'horiz_sd_flma', 'num_persistent_components_flma', 'vert_spread_flma',
                         'num_delaunay_edges_flma']
drop_cols = drop_cols + ['vert_spread_tbv', 'min_alpha2_betti_1_tbv', 'horiz_spread_tbv', 'num_delaunay_edges_tbv',
                         'inter_layer_bottleneck_avg_tbv']

# Retain 25 regressors

new_data = data.drop(columns=drop_cols)
new_data_normed = normed_all.drop(columns=drop_cols)

new_data_corr = new_data.corr()

new_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_parsimonious.csv')
new_data_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_normed_parsimonious.csv')

def get_high_acc_data(tol=0, save_data = False):
    """
    Truncates dataset below accuracy threshold tol and performs PCA and regression analysis. Saves datasets
    (including PCA-transformed data) to /pandas_dataframes directory.

    :param tol: accuracy threshold
    :param save_data: whether or not to save the dataframes.
    :return: None
    """
    all_data = pd.read_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_normed_parsimonious.csv', index_col=0)

    high_acc_data = all_data.where(all_data['final_accuracy'] > tol).dropna()
    print('Shape of new data: {}'.format(high_acc_data.shape))

    high_acc_means = high_acc_data.mean()
    high_acc_stds = high_acc_data.std()

    high_acc_normed = (high_acc_data - high_acc_means)/high_acc_stds
    if save_data:
        high_acc_data.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_{}_parsimonious.csv'.format(tol))
        high_acc_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_{}_normed_parsimonious.csv'.format(tol))

    y = high_acc_data['final_accuracy']
    X = high_acc_data.drop(columns=['final_accuracy'])
    X_norm = high_acc_normed.drop(columns=['final_accuracy'])

    lr = LinearRegression()

    lr.fit(X, y)

    print('R^2 for non-normalized data with acc > {}: {}'.format(tol, lr.score(X, y)))

    pca = PCA()
    pca.fit(X_norm)
    print('Principal component explained variance ratios: {}'.format(pca.explained_variance_ratio_))


for tol in np.linspace(0.0, 0.1, 21):

    get_high_acc_data(tol)

# Set accuracy tolerance to 0.055

high_acc = new_data.where(new_data['final_accuracy'] > 0.055).dropna()
high_acc_normed = new_data_normed.where(new_data_normed['final_accuracy'] > 0.055).dropna()

high_acc.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_{}_parsimonious.csv'.format(0.055))
high_acc_normed.to_csv('/Users/jhamer90811/PycharmProjects/MINERvA_NOvA_network_analysis/pandas_dataframes/minerva_complex_{}_normed_parsimonious.csv'.format(tol))

# Plot principal components to look for patterns.
pca_2 = PCA(2)
pca_3 = PCA(3)

X = high_acc_normed.drop(columns=['final_accuracy'])
y_acc = high_acc_normed.iloc[:, 0]

X_pca_2 = pca_2.fit_transform(X)
X_pca_3 = pca_3.fit_transform(X)

x = X_pca_2[:, 0]
y = X_pca_2[:, 1]

f = plt.figure(figsize=(8, 6))
ax = plt.axes()

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

ax.set_title('First Two Principal Components of Complex Attributes')

s = ax.scatter(x, y, c=y_acc)

plt.colorbar(s)

plt.show()

x = X_pca_3[:, 0]
y = X_pca_3[:, 1]
z = X_pca_3[:, 2]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

ax.set_title('First Three Principal Components of Complex Attributes')

s = ax.scatter(x, y, z, c=y_acc)

plt.colorbar(s)

plt.show()

min_acc = min(y_acc)
max_acc = max(y_acc)

num_bins = 6
bins = np.linspace(min_acc, max_acc+1e-8, num_bins + 1)
bins = [(i, bins[i], bins[i+1]) for i in range(num_bins)]
labels = [[i for i in range(num_bins) if bins[i][1] <= x < bins[i][2]][0] for x in y_acc]

pts = [[X_pca_2[i, :] for i in range(len(labels)) if labels[i] == k] for k in range(num_bins)]
xmin, xmax = min(X_pca_2[:,0]), max(X_pca_2[:,0])
ymin, ymax = min(X_pca_2[:,1]), max(X_pca_2[:,1])

for i in reversed(range(num_bins)):

    current_pts = pts[i:]
    acc = bins[i][1]
    num_points = 0
    f = plt.figure(figsize=(8, 6))
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

pts = [[X_pca_3[i, :] for i in range(len(labels)) if labels[i] == k] for k in range(num_bins)]
xmin, xmax = min(X_pca_3[:,0]), max(X_pca_3[:,0])
ymin, ymax = min(X_pca_3[:,1]), max(X_pca_3[:,1])
zmin, zmax = min(X_pca_3[:,2]), max(X_pca_3[:,2])

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

    for pt in reversed(current_pts):
        x, y, z = [p[0] for p in pt], [p[1] for p in pt], [p[2] for p in pt]
        ax.scatter(x, y, z, edgecolor='', s=10)
        num_points += len(pt)

    ax.text2D(0.05, 0.95, 'First 3 Principal Components; {} <= log(accuracy+1) <= {}; Number of Networks = {}'.format(np.round(acc, 4), np.round(max_acc, 4), num_points), transform=ax.transAxes)
    plt.show()

# Almost no patterns appear regarding accuracy

# Try to learn a manifold and call it a day...

pca = PCA()
X_pca = pca.fit_transform(X)

iso_2 = Isomap(n_components=2)
iso_3 = Isomap(n_components=3)

X_iso_2 = iso_2.fit_transform(X)
X_iso_3 = iso_3.fit_transform(X)

x = X_iso_2[:, 0]
y = X_iso_2[:, 1]

f = plt.figure(figsize=(8, 6))
ax = plt.axes()

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

ax.set_title('2D Isomap of Complex Attributes')

s = ax.scatter(x, y, c=y_acc)

plt.colorbar(s)

plt.show()

x = X_iso_3[:, 0]
y = X_iso_3[:, 1]
z = X_iso_3[:, 2]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

ax.set_title('3D Isomap of Complex Attributes')

s = ax.scatter(x, y, z, c=y_acc)

plt.colorbar(s)

plt.show()

tsne_2 = TSNE(2)
tsne_3 = TSNE(3)

X_tsne_2 = tsne_2.fit_transform(X)
X_tsne_3 = tsne_3.fit_transform(X)

x = X_tsne_2[:, 0]
y = X_tsne_2[:, 1]

f = plt.figure(figsize=(8, 6))
ax = plt.axes()

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

ax.set_title('2D t-SNE of Complex Attributes')

s = ax.scatter(x, y, c=y_acc)

plt.colorbar(s)

plt.show()

x = X_tsne_3[:, 0]
y = X_tsne_3[:, 1]
z = X_tsne_3[:, 2]

f = plt.figure(figsize=(8, 6))
ax = mplot3d.axes3d.Axes3D(f)

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))

ax.set_title('3D t-SNE of Complex Attributes')

s = ax.scatter(x, y, z, c=y_acc)

plt.colorbar(s)

plt.show()

percep = Perceptron()

y_acc_mean = y_acc.mean()
y_lab = [0 if a <= y_acc_mean else 1 for a in y_acc]

percep.fit(X_tsne_2, y_lab)

percep_predicted = percep.predict(X_tsne_2)

x = X_tsne_2[:, 0]
y = X_tsne_2[:, 1]

f = plt.figure(figsize=(8, 6))
ax = plt.axes()

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

ax.set_title('True Labels of t-SNE Embedding')

s = ax.scatter(x, y, c=y_lab)

plt.colorbar(s)

plt.show()

x = X_tsne_2[:, 0]
y = X_tsne_2[:, 1]

f = plt.figure(figsize=(8, 6))
ax = plt.axes()

ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

ax.set_title('Perceptron-Predicted Labels of t-SNE Embedding')

s = ax.scatter(x, y, c=percep_predicted)

plt.colorbar(s)

plt.show()

# Nothing very interesting...