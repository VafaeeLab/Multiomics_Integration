
Requirements:
    - Python 3.9
    - numpy
    - pandas
    - scipy
    - scikit-learn
    - tensorflow
    - matplotlib
    - seaborn

Usage:
    Run this script and provide the required file paths when prompted.


# ===============================
# Standard Library Imports
# ===============================
import os
import time
import random
import re
import json
import itertools
import datetim
from datetime import timedelta, datetime
import types
from collections import Counter
# ===============================
# Data Manipulation and Computation
# ===============================
import numpy as np
import pandas as pd

# ===============================
# Statistical and Mathematical Analysis
# ===============================
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import cut_tree, linkage, fcluster

# ===============================
# Machine Learning and Preprocessing
# ===============================
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances

# ===============================
# Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Graph Theory
# ===============================
import networkx as nx

# ===============================
# Deep Learning
# ===============================
import tensorflow as tf

# ===============================
# Custom Modules
# ===============================
from src.models.MLPNN import MLPNN
from src.models.MiMeNet import MiMeNet, tune_MiMeNet

# ===============================
# Global Variables and User Input
# ===============================
G1 = "CON"  # Group 1
G2 = "CIR"  # Group 2

# Prompt user for file paths or use a relative directory structure
input_path = input("Enter the path to the stool species CSV file: ")
output_path = input("Enter the path to the metabolomic CSV file: ")
met_annotation_path = input("Enter the path to the MET_Annotation CSV file: ")
diagnosis_path = input("Enter the path to the diagnosis CSV file: ")
metabolomic_h_path = input("Enter the path to the Metabolomic_H CSV file: ")

# ===============================
# Data Loading
# ===============================
micro_df = pd.read_csv(input_path)
metab_df = pd.read_csv(output_path)
metabolome_meta_df = pd.read_csv(met_annotation_path, index_col=0)
diagnosis = pd.read_csv(diagnosis_path, index_col=0)
metabolomic_h = pd.read_csv(metabolomic_h_path)

# Define the classes to extract
classes_to_extract = [G1, G2]

# Filter rows with labels G1 and G2 in metab_df
original_labels = metab_df['Label']
metab_df = metab_df[original_labels.isin(classes_to_extract)]

# Filter rows with labels G1 and G2 in micro_df
micro_df = micro_df[original_labels.isin(classes_to_extract)]

# Filter diagnosis to include only the selected classes
class_mask = diagnosis['Label'].isin(classes_to_extract)
diagnosis = diagnosis[class_mask]
samples = diagnosis.index

# ===============================
# Data Cleaning and Filtering
# ===============================
# Filter microbiome dataset: Remove columns with >90% zero values
columns_to_exclude_micro = micro_df.columns[(micro_df == 0).mean(axis=0) >= 0.9]
micro_df = micro_df.drop(columns=columns_to_exclude_micro)

# Filter metabolite dataset: Remove columns with >90% zero values
columns_to_exclude_metab = metab_df.columns[(metab_df == 0).mean(axis=0) >= 0.9]
metab_df = metab_df.drop(columns=columns_to_exclude_metab)

# Remove features with >50% missing values in metabolite dataset
missing_percentage = (metab_df.isnull().sum() / metab_df.shape[0]) * 100
threshold = 50  # Missing percentage threshold
selected_features = missing_percentage[missing_percentage <= threshold].index
metab_df = metab_df[selected_features]

# Add back the 'Label' column
metab_df['Label'] = original_labels
micro_df['Label'] = original_labels


# =======================================
#Define different Normalization Methods  
# ========================================

# Scale normalization function with NaN handling

def scale_normalization(df):
    # Convert all columns to numeric, forcing non-convertible values to NaN
    labels = metab_df['Label']
    df = df.apply(pd.to_numeric, errors='coerce')

    # Exclude the 'Label' column from normalization
    features = df.columns[df.columns != 'Label']

    # Scale normalization for each feature
    for feature in features:
        non_nan_indices = ~df[feature].isna()  # Find non-NaN indices
        mean_value = df.loc[non_nan_indices, feature].mean()  # Compute mean excluding NaNs
        std_value = df.loc[non_nan_indices, feature].std()  # Compute std deviation excluding NaNs
        df.loc[non_nan_indices, feature] = (df.loc[non_nan_indices, feature] - mean_value) / std_value

    df['Label'] = labels

    return df

# Define a function for count per million (CPM) and log transformation

def cpm_and_log_transformation(df):
    # Convert all columns to numeric, forcing non-convertible values to NaN
    labels = metab_df['Label']

    # Exclude the 'Label' column from transformation
    features = df.columns[df.columns != 'Label']
    df = df.apply(pd.to_numeric, errors='coerce')

    # Compute counts per million (CPM)
    cpm_df = df.copy()
    cpm_df[features] = cpm_df[features].apply(lambda x: (x / x.sum()) * 1e6, axis=1)

    # Apply log transformation
    log_cpm_df = cpm_df.copy()
    log_cpm_df[features] = np.log1p(log_cpm_df[features] + 0.01)
    log_cpm_df['Label'] = labels

    return log_cpm_df

# Apply scale normalization on metab_df
metab_df = cpm_and_log_transformation(metab_df)

# Apply count per million (CPM) and log transformation on micro_df
micro_df = cpm_and_log_transformation(micro_df)


# ========================================
# knn Imputation
# ========================================

metab_df_nolabel = metab_df.drop(columns=['Label'])
metab_comp_miss_df=metab_df_nolabel
# Create a KNNImputer
imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors
# Impute missing values using KNN
imputed_data = imputer.fit_transform(metab_comp_miss_df)
# Convert the imputed data back to a DataFrame
metab_comp_df = pd.DataFrame(imputed_data, columns=metab_comp_miss_df.columns)

metab_comp_df.index=metab_comp_miss_df.index
micro_comp_df=micro_df.drop(columns=['Label'])

# ========================================
# MICE Imputation
# ========================================

# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# # Assuming 'metab_df' is already defined
# metab_df_nolabel = metab_df.drop(columns=['Label'])

# metab_comp_miss_df = metab_df_nolabel

# # Create an IterativeImputer (MICE)
# imputer = IterativeImputer(random_state=42)  # You can adjust parameters as needed

# # Impute missing values using MICE
# imputed_data = imputer.fit_transform(metab_comp_miss_df)

# # Convert the imputed data back to a DataFrame
# metab_comp_df = pd.DataFrame(imputed_data, columns=metab_comp_miss_df.columns)

# metab_comp_df.index = metab_comp_miss_df.index
#micro_comp_df=micro_df.drop(columns=['Label'])

AA=metab_comp_df
BB=micro_comp_df

# =====================================================
#Filteration based on differential expression analysis
# =====================================================
# Load your data
microbes = micro_comp_df
metabolites = metab_comp_df
labels = diagnosis

# Define a function for differential expression analysis
def differential_expression(data, labels, p_value_threshold=0.15):
    condition_1 = data[labels == G1]
    condition_2 = data[labels == G2]
    
    p_values = []
    for feature in data.columns:
        stat, p_value = ttest_ind(condition_1[feature], condition_2[feature], nan_policy='omit')
        p_values.append(p_value)
    
    p_values = np.array(p_values)
    selected_features = data.columns[p_values < p_value_threshold]
    
    return selected_features

# Define a function for fold change filtering
def fold_change_filter(data, labels, fold_change_threshold=1.5):
    condition_1_mean = data[labels == G1].mean()
    condition_2_mean = data[labels == G2].mean()
    
    fold_changes = abs(condition_1_mean / condition_2_mean)
    selected_features = data.columns[fold_changes < fold_change_threshold]
    
    return selected_features

# Apply differential expression analysis
selected_microbes_de = differential_expression(microbes, labels['Label'], p_value_threshold=0.15)
selected_metabolites_de = differential_expression(metabolites, labels['Label'], p_value_threshold=0.15)

# Apply fold change filtering
selected_microbes_fc = fold_change_filter(microbes, labels['Label'], fold_change_threshold=1.5)
selected_metabolites_fc = fold_change_filter(metabolites, labels['Label'], fold_change_threshold=1.5)

# Combine the selected features from both methods
final_selected_microbes = set(selected_microbes_de).intersection(set(selected_microbes_fc))
final_selected_metabolites = set(selected_metabolites_de).intersection(set(selected_metabolites_fc))

# Filter the original dataframes to keep only the selected features
filtered_microbes = microbes[list(selected_microbes_de)]
filtered_metabolites = metabolites[list(selected_metabolites_de)]

micro_comp_df=microbes[list(selected_microbes_de)]
metab_comp_df=metabolites[list(selected_metabolites_de)]

CC=metab_comp_df
DD=micro_comp_df


# MLP architecture
num_run_cv = 10
num_cv = 10
tuned = True

print("Loading network parameters...")
try:
    # Prompt the user to specify the file path
    file_path = input("Please specify the full path to the network parameters file (e.g., C:/path/to/network_parameters.txt): ").strip()

    with open(file_path, "r") as infile:
        params = json.load(infile)
        num_layer = params["num_layer"]
        layer_nodes = params["layer_nodes"]
        l1 = params["l1"]
        l2 = params["l2"]
        dropout = params["dropout"]
        learning_rate = params["lr"]
        tuned = True
        print("Loaded network parameters successfully.")
except FileNotFoundError:
    print(f"Error: The file at '{file_path}' was not found. Please check the path and try again.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file at '{file_path}'. Ensure the file is in the correct format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

score_matrices = []
print("Performing %d runs of %d-fold cross-validation" % (num_run_cv, num_cv))
cv_start_time = time.time()
tune_run_time = 0

micro = micro_comp_df.values
metab = metab_comp_df.values
    
dirName =folder_path
try:
    os.mkdir(dirName)
except FileExistsError:
    pass

for run in range(0,num_run_cv):

    # Set up output directory for CV runs
    dirName =folder_path + str(run)
    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass

    np.random.seed(42)
    random.seed(42)

    # Set up CV partitions
    kfold = KFold(n_splits=num_cv, shuffle=True, random_state=42)
    cv = 0

    for train_index, test_index in kfold.split(samples):

        # Set up output directory for CV partition run
        dirName = folder_path + str(run) + '/' + str(cv)
        try:
            os.mkdir(dirName)
        except FileExistsError:
            pass

        # Partition data into training and test sets
        train_micro, test_micro = micro[train_index], micro[test_index]
        train_metab, test_metab = metab[train_index], metab[test_index]
        train_samples, test_samples = samples[train_index], samples[test_index]

        # Store training and test set partitioning
        train_microbe_df = pd.DataFrame(data=train_micro, index=train_samples, columns=micro_comp_df.columns)
        test_microbe_df = pd.DataFrame(data=test_micro, index=test_samples, columns=micro_comp_df.columns)
        train_metab_df = pd.DataFrame(data=train_metab, index=train_samples, columns=metab_comp_df.columns)
        test_metab_df = pd.DataFrame(data=test_metab, index=test_samples, columns=metab_comp_df.columns)

        train_microbe_df.to_csv(dirName + "/train_microbes.csv")
        test_microbe_df.to_csv(dirName + "/test_microbes.csv")
        train_metab_df.to_csv(dirName + "/train_metabolites.csv")
        test_metab_df.to_csv(dirName + "/test_metabolites.csv")

        # Log transform data
        train_micro = np.log(train_micro + 1)
        test_micro = np.log(test_micro + 1)
        #train_metab = np.log(train_metab + 1)
        #test_metab = np.log(test_metab + 1)

        # Scale data before neural network training
        micro_scaler = StandardScaler().fit(train_micro)
        train_micro = micro_scaler.transform(train_micro)
        test_micro = micro_scaler.transform(test_micro)

        metab_scaler = StandardScaler().fit(train_metab)
        train_metab = metab_scaler.transform(train_metab)
        test_metab = metab_scaler.transform(test_metab)

        # Aggregate paired microbiome and metabolomic data
        train = (train_micro, train_metab)
        test = (test_micro, test_metab)

        # Tune hyperparameters if first partition
        if tuned == False:
            print("Tuning parameters...")
            tuned = True
            params = tune_MiMeNet(train)
            l1 = params['l1']
            l2 = params['l2']
            num_layer=params['num_layer']
            layer_nodes=params['layer_nodes']
            learning_rate = params["lr"]
            dropout=params['dropout']
            with open('results/network_parameters.txt', 'w') as outfile:
                json.dump(params, outfile)

            tune_run_time = time.time() - tune_start_time
            print("Tuning run time: " + (str(datetime.timedelta(seconds=(tune_run_time)))))

        print("Run: %02d\t\tFold: %02d" % (run + 1, cv + 1), end="\r")

        # Construct Neural Network Model
        model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=l1, l2=l2,
                            num_layer=num_layer, layer_nodes=layer_nodes, dropout=dropout)

        #Train Neural Network Model
        model.train(train)

        # Predict on test set
        p = model.test(test)

        inv_p = metab_scaler.inverse_transform(p)

        #####
        inv_p = np.exp(inv_p) - 1
        inv_p = inv_p/np.sum(inv_p)
        #####

        score_matrices.append(model.get_scores())
        prediction_df = pd.DataFrame(data=inv_p, index=test_samples, columns=metab_comp_df.columns)
        score_matrix_df = pd.DataFrame(data=model.get_scores(), index=micro_comp_df.columns, columns=metab_comp_df.columns)

        prediction_df.to_csv(dirName + "/prediction.csv")
        score_matrix_df.to_csv(dirName + "/score_matrix.csv")
        model.destroy()
        tf.keras.backend.clear_session()

        cv += 1


print("\nCV run time: " + str(datetime.timedelta(seconds=(time.time() - cv_start_time - tune_run_time))))



print("\nCalculating correlations for cross-validated evaluation...")

correlation_cv_df = pd.DataFrame(index=metab_comp_df.columns)

for run in range(num_run_cv):
    preds = pd.concat([pd.read_csv(folder_path+ str(run)+ '/' + str(cv) + "/prediction.csv",
                                    index_col=0) for cv in range(0, num_cv)])
    y = pd.concat([pd.read_csv(folder_path + str(run)+ '/' + str(cv) + "/test_metabolites.csv",
                                index_col=0) for cv in range(0, num_cv)])

    cor = y.corrwith(preds, method="spearman")

    correlation_cv_df["Run_"+str(run)] = cor.loc[correlation_cv_df.index]

correlation_cv_df["Mean"] = correlation_cv_df.mean(axis=1)
correlation_cv_df = correlation_cv_df.sort_values("Mean", ascending=False)
correlation_cv_df.to_csv(folder_path+"/cv_correlations.csv")

fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(111)
sns.distplot(correlation_cv_df["Mean"], kde=True)

plt.title("Prediction Correlation")
plt.ylabel("Frequency")
plt.xlabel("Spearman Correlation")
plt.text(0.1, 0.9,"Mean: %.3f"% np.mean(correlation_cv_df.values),
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
cv_correlation_distribution_path = os.path.join(folder_path, "SCCs Predicted vs Test Metabolome.png")
plt.savefig(cv_correlation_distribution_path)
cv_correlation_distribution_path = os.path.join(folder_path, "SCCs Predicted vs Test Metabolome.pdf")
plt.savefig(cv_correlation_distribution_path)

plt.savefig(folder_path+"/SCCs Predicted vs Test Metabolome.png")
print("Mean correlation: %f" % np.mean(correlation_cv_df.values))

    

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# MLP architecture
print("Generating background using 100 10-fold cross-validated runs of shuffled data...")
num_bg= 10
n_split= 10
bg_preds = []
bg_truth = []
bg_scores = []
bg_start_time = time.time()
for run in range(0,num_bg):
    preds = []
    truth = []
    score_matrix = []

    micro = micro_comp_df.values
    metab = metab_comp_df.values

    np.random.shuffle(micro)
    np.random.shuffle(metab)

    kfold = KFold(n_splits=n_split,shuffle=True, random_state=42)
    cv=0
    for train_index, test_index in kfold.split(micro):
        print("Run: %02d\t\tFold:%02d" % (run + 1, cv + 1), end="\r")
        train_micro, test_micro  = micro[train_index], micro[test_index]
        train_metab, test_metab = metab[train_index], metab[test_index]


        train_micro = np.log(train_micro + 1)
        test_micro = np.log(test_micro + 1)
        #train_metab = np.log(train_metab + 1)
        #test_metab = np.log(test_metab + 1)

        # Scale data before neural network training
        micro_scaler = StandardScaler().fit(train_micro)
        train_micro = micro_scaler.transform(train_micro)
        test_micro = micro_scaler.transform(test_micro)

        metab_scaler = StandardScaler().fit(train_metab)
        train_metab = metab_scaler.transform(train_metab)
        test_metab = metab_scaler.transform(test_metab)

        train = (train_micro, train_metab)
        test = (test_micro, test_metab)

        model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=l1, l2=l2,
                            num_layer=num_layer, layer_nodes=layer_nodes, dropout=dropout)

        model.train(train)
        p = model.test(test)

        preds = list(preds) + list(p)
        truth = list(truth) + list(test_metab)
        score_matrix.append(model.get_scores())

        model.destroy()
        tf.keras.backend.clear_session()
        cv+=1

    bg_preds.append(preds)
    bg_truth.append(truth)
    bg_scores.append(score_matrix)

# Set up output directory for training on full dataset
dirName = 'results/Full'

try:
    os.mkdir(dirName)
except FileExistsError:
    pass

microbe_cluster_matrix_list = []
metabolite_cluster_matrix_list = []



dirName = folder_path
try:
    os.mkdir(dirName)
except FileExistsError:
    pass

bg_preds = np.array(bg_preds)
bg_truth = np.array(bg_truth)
bg_scores = np.array(bg_scores)
bg_scores_mean = np.mean(np.array(bg_scores), axis=1)

outfile = open(dirName + "bg_preds.pkl", "wb")
pickle.dump(np.array(bg_preds), outfile)
outfile.close()

outfile = open(dirName + "bg_truth.pkl", "wb")
pickle.dump(np.array(bg_truth), outfile)
outfile.close()

outfile = open(dirName + "bg_scores_mean.pkl", "wb")
pickle.dump(bg_scores_mean, outfile)
outfile.close()

bg_corr = []

for i in range(0, bg_preds.shape[0]):
    for j in range(0,bg_preds.shape[-1]):
        p_vec = bg_preds[i,:,j]
        m_vec = bg_truth[i,:,j]
        cor = spearmanr(p_vec, m_vec)
        bg_corr.append(cor[0])

outfile = open(dirName + "bg_correlations.pkl", "wb")
pickle.dump(np.array(bg_corr), outfile)
outfile.close()


cutoff_rho = np.quantile(bg_corr, 0.95)

print("The correlation cutoff is %.3f" % cutoff_rho)
print("%d of %d metabolites are significantly correlated" % (sum(correlation_cv_df["Mean"].values > cutoff_rho),
                                                             len(correlation_cv_df["Mean"].values)))

sig_metabolites = correlation_cv_df.index[correlation_cv_df["Mean"].values > cutoff_rho]
WPM=sig_metabolites.shape[0]

annotated_metabolites = np.intersect1d(correlation_cv_df.index.values, metabolome_meta_df.index.values)
sig_metabolites_annotated = annotated_metabolites[correlation_cv_df.loc[annotated_metabolites, "Mean"].values > cutoff_rho]

print("%d of %d annotated metabolites are significantly correlated" % (len(sig_metabolites_annotated), len(annotated_metabolites)))

barplot_df = pd.DataFrame(data={"Compound Name":metabolome_meta_df.loc[sig_metabolites_annotated, "Compound Name"].values,
                                "Spearman Correlation": correlation_cv_df.loc[sig_metabolites_annotated, "Mean"].values},
                              index=metabolome_meta_df.loc[sig_metabolites_annotated, "Compound Name"].values)

barplot_df["Compound Name"] = [x.strip().capitalize() for x in barplot_df["Compound Name"].values]

fig = plt.figure(figsize=(8,16), dpi=300)
ax = fig.add_subplot(111)
sns.barplot(x="Spearman Correlation", y='Compound Name', ci=None,
         data=barplot_df.groupby(barplot_df.index).max().sort_values(by="Spearman Correlation", ascending=False).head(20))
ax.tick_params(axis='y', which='major', labelsize=12)
plt.xlabel("Spearman Correlation", fontsize=10)
plt.tight_layout()
fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(111)
sns.barplot(x="Spearman Correlation", y='Compound Name',
                data=barplot_df.groupby(barplot_df.index).max().sort_values(by="Spearman Correlation", ascending=False).head(20))
plt.savefig("results/top_correlated_metabolites.png")


top_correlated_metabolites_path = os.path.join(folder_path, "Top20_Significantly Correlated Metabolites.png")
plt.savefig(top_correlated_metabolites_path)
top_correlated_metabolites_path = os.path.join(folder_path, "Top20_Significantly Correlated Metabolites.pdf")
plt.savefig(top_correlated_metabolites_path)

sig_metabolites_csv_path = os.path.join(folder_path, "significant_metabolites.csv")
sig_metabolites_df = pd.DataFrame(sig_metabolites, columns=['Significant_Metabolites'])
sig_metabolites_df.to_csv(sig_metabolites_csv_path, index=False)

    
# Load the data containing the mappings
metabolomic_path = input("Enter the path to the Metabolomic CSV file: ")
metabolomic_h_path = input("Enter the path to the Metabolomic_H CSV file: ")
metabolomic = pd.read_csv(metabolomic_path)
metabolomic_h = pd.read_csv(metabolomic_h_path)

# Set your cutoff value
cutoff_rho = 0.5  # Adjust as needed

# Map feature names to real names using Metabolomic_H.csv
column_names_to_search = annotated_metabolites
column_indices = [metabolomic.columns.get_loc(name) for name in column_names_to_search]
corresponding_column_names = [metabolomic_h.columns[i] for i in column_indices]

# Create a mapping from feature name to real name
real_name_mapping = dict(zip(annotated_metabolites, corresponding_column_names))

# Replace Compound Name with real names in the barplot dataframe
barplot_df = pd.DataFrame(data={
    "Compound Name": [real_name_mapping[name].strip().capitalize() for name in annotated_metabolites],
    "Spearman Correlation": correlation_cv_df.loc[annotated_metabolites, "Mean"].values
})

# Plot the barplot
fig = plt.figure(figsize=(16, 8), dpi=300)
ax = fig.add_subplot(111)
sns.barplot(
    x="Spearman Correlation", 
    y="Compound Name", 
    ci=None,
    data=barplot_df.groupby("Compound Name").max().sort_values(by="Spearman Correlation", ascending=False).head(20)
)
ax.tick_params(axis='y', which='major', labelsize=12)
plt.xlabel("Spearman Correlation", fontsize=10)
plt.tight_layout()

# Save the plot
top_correlated_metabolites_path = os.path.join(folder_path, "top_correlated_metabolites_h.png")
plt.savefig(top_correlated_metabolites_path)
top_correlated_metabolites_path = os.path.join(folder_path, "top_correlated_metabolites_h.pdf")
plt.savefig(top_correlated_metabolites_path)
# Display the plot
plt.show()



#######################################################
# Mapping Significant Metabolites to Original Names
#######################################################


column_names_to_search = sig_metabolites_df.index

# Find the column indices of the specified column names in Metabolomic.csv
column_indices = [metabolomic.columns.get_loc(name) for name in column_names_to_search]

# Find the corresponding column names in Metabolomic_H.csv
corresponding_column_names = [metabolomic_h.columns[i] for i in column_indices]

# Print the corresponding column names
for original, corresponding in zip(column_names_to_search, corresponding_column_names):
    print(f"The corresponding column name for '{original}' in Metabolomic_H.csv is '{corresponding}'")

# If you want to save the mapping to a file, you can write to a CSV or text file
mapping_df = pd.DataFrame({
    'Metabolomic': column_names_to_search,
    'Metabolomic_H': corresponding_column_names
})

mapping_df_path = os.path.join(folder_path, "significant_metabolites_header.csv")
mapping_df.to_csv(mapping_df_path)



#######################################################
# Identify microbes with significant interaction scores
#######################################################

# Calculate the mean score matrix by averaging across all score matrices
mean_score_matrix = np.mean(np.array(score_matrices), axis=0)

# Filter the mean score matrix to keep only columns that correspond to significant metabolites
reduced_mean_score_matrix = mean_score_matrix[:,[x in sig_metabolites for x in correlation_cv_df.index]]
reduced_bg_score_matrix = bg_scores_mean[:,:,[x in sig_metabolites for x in correlation_cv_df.index]]

sig_edge_matrix = np.zeros(reduced_mean_score_matrix.shape)

for mic in range(reduced_mean_score_matrix.shape[0]):
    for met in range(reduced_mean_score_matrix.shape[1]):
        sig_cutoff = np.abs(np.quantile(reduced_bg_score_matrix[:,mic,met], 0.975))         # Set the significance cutoff at the 97.5th percentile of the background score distribution
        if np.abs(reduced_mean_score_matrix[mic,met]) > sig_cutoff:
            sig_edge_matrix[mic,met]=1


sig_microbes = micro_comp_df.columns[np.sum(sig_edge_matrix, axis=1)> 0.01 * len(sig_metabolites)]

NSM=sig_microbes.shape[0]

# Convert the sig_microbes to a DataFrame to save it as a CSV file
sig_microbes_csv_path = os.path.join(folder_path, "significant_microbes.csv")
sig_microbes_df = pd.DataFrame(sig_microbes, columns=['Significant_Microbes'])
sig_microbes_df.to_csv(sig_microbes_csv_path, index=False)



###################################################
# Compare Correlation Distributions
###################################################
# Create the figure and axis for plotting
fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(111)

# Plot the observed correlation distribution vs. the background distribution
sns.distplot(correlation_cv_df["Mean"].values, bins=20, label="Observed")
sns.distplot(bg_corr, label="Background")
plt.axvline(x=cutoff_rho, color="red", lw=2, label="95% Cutoff")
plt.axvspan(cutoff_rho, 1.0, alpha=0.2, color='gray')

plt.title("Correlation Distributions", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.xlabel("Spearman Correlation", fontsize=16)
plt.xlim(-1,1)
plt.text(0.85, 0.85,"Significant Region",
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes)
plt.legend()
plt.savefig("results/cv_bg_correlation_distributions.png")

cv_bg_correlation_distributions = os.path.join(folder_path, "Distributions of SCCs in Background and Observed_Metabolite.png")
plt.savefig(cv_bg_correlation_distributions)
cv_bg_correlation_distributions = os.path.join(folder_path, "Distributions of SCCs in Background and Observed_Metabolite.pdf")
plt.savefig(cv_bg_correlation_distributions)

# Compute the score matrix and filter for significant microbes and metabolites
score_matrix_df = pd.DataFrame(np.mean(score_matrices, axis=0), index=micro_comp_df.columns,
                            columns=metab_comp_df.columns)

reduced_score_df = score_matrix_df.loc[sig_microbes,sig_metabolites]

# Convert scores into binary based on the significance cutoff
binary_score_df = pd.DataFrame(np.clip(reduced_score_df/sig_cutoff, -1, 1), index=sig_microbes,
                               columns= sig_metabolites)

# Save the score matrices as CSV files
score_matrix_df_path = os.path.join(folder_path, "score_matrix_df.csv")
score_matrix_df.to_csv(score_matrix_df_path, index=False)
binary_score_df_path = os.path.join(folder_path, "binary_score_df.csv")
binary_score_df.to_csv(binary_score_df_path, index=False)



mean_score_matrix_met=np.mean(correlation_cv_df["Mean"].values)
###################################################
# Compute Number of Microbial Modules
###################################################

mic_connectivity_matrices = {}
num_run=10
for i in range(2,20):
    mic_connectivity_matrices[i] = np.zeros((len(sig_microbes), len(sig_microbes)))

# For each score matrix, compute hierarchical clustering and update connectivity matrices
for s in score_matrices:
    mic_linkage_list = shc.linkage(np.clip(s[[x in sig_microbes for x in micro_comp_df.columns],:][:,[x in sig_metabolites for x in metab_comp_df.columns]]/sig_cutoff, -1,1), method='complete')
    for i in range(2,20):
        # Cut the hierarchical clustering tree into 'i' clusters and one-hot encode the clusters
        microbe_clusters = np.array(cut_tree(mic_linkage_list, n_clusters=i)).reshape(-1)
        one_hot_matrix = np.zeros((len(sig_microbes), i))
        for m in range(len(microbe_clusters)):
            one_hot_matrix[m, microbe_clusters[m]] = 1
        # Calculate the connectivity matrix and accumulate it for consensus calculation
        mic_connectivity_matrix = np.matmul(one_hot_matrix, np.transpose(one_hot_matrix))
        mic_connectivity_matrices[i] += mic_connectivity_matrix
        
# Plot the consensus CDF for different numbers of clusters
fig = plt.figure(figsize=(8,4), dpi=300)
plt.subplot(1,2,1)
area_x = []
area_y = []
for i in range(2,20):
    consensus_matrix = mic_connectivity_matrices[i]/(num_run_cv * num_run)  # Average the connectivity matrices over multiple runs
    n = consensus_matrix.shape[0]
    consensus_cdf_x = []
    consensus_cdf_y = []
    area_x.append(int(i))
    
    # Compute the area under the CDF for the consensus matrix
    prev_y = 0
    prev_x = 0
    area = 0
    for j in range(0,101):
        x = float(j)/100.0
        y = sum(sum(consensus_matrix <= x))/(n*(n-1))
        consensus_cdf_x.append(x)
        consensus_cdf_y.append(y)
        area += (x-prev_x) * (y)
        prev_x = x
    area_y.append(area)
    
    # Plot the consensus CDF for the current number of clusters
    plt.plot(consensus_cdf_x, consensus_cdf_y, label=str(i) + " Clusters")

plt.xlabel("Consensus Index Value")
plt.ylabel("CDF")
plt.legend()

# Calculate the relative increase in area under the CDF to determine the optimal number of clusters
dk = []
for a in range(len(area_x)):
    if area_x[a] == 2:
        dk.append(area_y[a])
    else:
        dk.append((area_y[a] - area_y[a-1])/area_y[a-1])
        
plt.title("microbiome clusters", fontsize=16)

# Plot the relative increase in area under CDF for different numbers of clusters
plt.subplot(1,2,2)
plt.plot(area_x, dk, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Relative Increase of Area under CDF")
plt.axhline(0.025, linewidth=2, color='r')

# Identify the optimal number of microbe clusters based on a threshold
num_microbiome_clusters = np.max(np.array(area_x)[np.array(dk) > 0.02])
print("Using %d Microbe Clusters" % num_microbiome_clusters)
# Save the plot of the cluster consensus
plt.savefig("results/microbe_cluster_consensus.png")
microbe_cluster_consensus = os.path.join(folder_path, "Determination of optimal number of microbes clusters.png")
plt.savefig(microbe_cluster_consensus)
microbe_cluster_consensus = os.path.join(folder_path, "Determination of optimal number of microbes clusters.pdf")
plt.savefig(microbe_cluster_consensus)

###################################################
# Compute Number of Metabolic Modules
###################################################
# Initialize a dictionary to store connectivity matrices for different numbers of clusters
met_connectivity_matrices = {}

for i in range(2,20):
    # Create zero matrices to store the connectivity values for metabolites
    met_connectivity_matrices[i] = np.zeros((len(sig_metabolites), len(sig_metabolites)))
# Perform clustering for each score matrix and update the connectivity matrices
for s in score_matrices:
    # Transpose and normalize the score matrix for metabolites and compute hierarchical clustering
    met_linkage_list = shc.linkage(np.transpose(np.clip(s[[x in sig_microbes for x in micro_comp_df.columns],:][:,[x in sig_metabolites for x in metab_comp_df.columns]]/sig_cutoff, -1,1)), method='complete')

    for i in range(2,20):
        metabolite_clusters  = np.array(cut_tree(met_linkage_list, n_clusters=i)).reshape(-1)
        one_hot_matrix = np.zeros((len(sig_metabolites), i))
        for m in range(len(metabolite_clusters )):
            one_hot_matrix[m, metabolite_clusters [m]] = 1
        met_connectivity_matrix  = np.matmul(one_hot_matrix, np.transpose(one_hot_matrix))
        met_connectivity_matrices[i] += met_connectivity_matrix

fig = plt.figure(figsize=(8,4), dpi=300)
plt.subplot(1,2,1)
area_x = []
area_y = []
for i in range(2,20):
    # Compute the consensus matrix by averaging connectivity matrices over multiple runs
    consensus_matrix = met_connectivity_matrices[i]/(num_run_cv * num_run)
    n = consensus_matrix.shape[0]
    consensus_cdf_x = []
    consensus_cdf_y = []
    area_x.append(int(i))
    # Calculate the area under the CDF for the consensus matrix
    prev_y = 0
    prev_x = 0
    area = 0
    for j in range(0,101):
        x = float(j)/100.0
        y = sum(sum(consensus_matrix <= x))/(n*(n-1))
        consensus_cdf_x.append(x)
        consensus_cdf_y.append(y)
        area += (x-prev_x) * (y)
        prev_x = x
    area_y.append(area)
    # Plot the CDF for the current number of clusters
    plt.plot(consensus_cdf_x, consensus_cdf_y, label=str(i) + " Clusters")

plt.xlabel("Consensus Index Value")
plt.ylabel("CDF")
plt.legend()

# Calculate the relative increase in area under the CDF for identifying optimal clusters
dk = []
for a in range(len(area_x)):
    if area_x[a] == 2:
        dk.append(area_y[a])
    else:
        dk.append((area_y[a] - area_y[a-1])/area_y[a-1])


plt.title("metabolite clusters", fontsize=16)
# Plot the relative increase in the area under CDF for different numbers of clusters
plt.subplot(1,2,2)
plt.plot(area_x, dk, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Relative Increase of Area under CDF")

plt.axhline(0.025, linewidth=2, color='r')

# Determine the optimal number of metabolite clusters based on the relative increase
num_metabolite_clusters  = np.max(np.array(area_x)[np.array(dk) > 0.02])
print("Using %d Metabolite Clusters" % num_metabolite_clusters)
# Save the consensus plot for metabolite clusters
plt.savefig("results/metabolite_cluster_consensus.png")

# Define the path for saving the cluster consensus image
metabolite_cluster_consensus = os.path.join(folder_path, "Determination of optimal number of metabolite clusters.png")
plt.savefig(metabolite_cluster_consensus)
metabolite_cluster_consensus = os.path.join(folder_path, "Determination of optimal number of metabolite clusters.pdf")
plt.savefig(metabolite_cluster_consensus)

###################################################
# Bicluster Interaction Matrix
###################################################

## Hierarchical Clustering

np.random.seed(42)
# Perform hierarchical clustering on both the microbiome and metabolite data
# Microbe tree: hierarchical clustering on the rows (microbes)
# Metabolite tree: hierarchical clustering on the columns (metabolites)
microbe_tree = shc.linkage(binary_score_df.values, method='complete')
metabolite_tree = shc.linkage(np.transpose(binary_score_df.values), method='complete')


# Cut the hierarchical tree to form clusters
# 'num_metabolite_clusters' and 'num_microbiome_clusters' represent the number of clusters for metabolites and microbes, respectively
metabolite_clusters = np.array(cut_tree(metabolite_tree, n_clusters=num_metabolite_clusters)).reshape(-1)
microbe_clusters = np.array(cut_tree(microbe_tree, n_clusters=num_microbiome_clusters)).reshape(-1)

metab_col_scale = 1/(num_metabolite_clusters-1)
micro_col_scale = 1/(num_microbiome_clusters-1)

wild_card_micro = np.random.uniform(0,1,num_microbiome_clusters)
wild_card_metab = np.random.uniform(0,1,num_metabolite_clusters)

# Generate a list of colors for each cluster, combining random and scaled components
micro_colors = [(wild_card_micro[x],1-x*micro_col_scale,x*micro_col_scale) for x in microbe_clusters]
# Metabolite colors: similar logic as for microbes, but the colors are different
metab_colors = [(1-x*metab_col_scale,wild_card_metab[x],x*metab_col_scale) for x in metabolite_clusters]

# Create a clustered heatmap using seaborn's clustermap
# binary_score_df is the input data (interaction score matrix)
sns.clustermap(binary_score_df, method="complete", row_colors = micro_colors, col_colors=metab_colors, row_linkage=microbe_tree,
              col_linkage=metabolite_tree, cmap = "coolwarm", figsize=(8,8), cbar_pos=(0.05, 0.88, 0.025, 0.10))

# Save the clustermap as a PNG file with high resolution
plt.savefig("results/clustermap.png", dpi=300)
# Save the clustermap as a PDF file with higher resolution
plt.savefig("results/clustermap3.pdf", dpi=600)

# Define the file path for saving the clustermap image in another directory
clustermap = os.path.join(folder_path, "Heatmap of Microbe-Metabolite Interactions.png")
plt.savefig(clustermap)
clustermap = os.path.join(folder_path, "Heatmap of Microbe-Metabolite Interactions.pdf")
plt.savefig(clustermap)

# Save the reduced score DataFrame to a CSV file
reduced_score_df.to_csv("results/CV/interaction_score_matrix.csv")



# Initialize matrices to store cluster relationships for microbes and metabolites
micro_cluster_matrix = np.zeros((reduced_score_df.values.shape[0], reduced_score_df.values.shape[0]))
metab_cluster_matrix = np.zeros((reduced_score_df.values.shape[1], reduced_score_df.values.shape[1]))

# Fill the microbe cluster matrix based on cluster membership
for m in range(0, len(microbe_clusters)):
  for n in range(m, len(microbe_clusters)):
      if microbe_clusters[m] == microbe_clusters[n]:
          micro_cluster_matrix[m,n] = 1
          micro_cluster_matrix[n,m] = 1
# Fill the metabolite cluster matrix similarly based on cluster membership
for m in range(0, len(metabolite_clusters)):
  for n in range(m, len(metabolite_clusters)):
      if metabolite_clusters[m] == metabolite_clusters[n]:
          metab_cluster_matrix[m,n] = 1
          metab_cluster_matrix[n,m] = 1
# Save the microbe cluster matrix as a CSV file
pd.DataFrame(data=micro_cluster_matrix, index=reduced_score_df.index,
              columns=reduced_score_df.index).to_csv("results/CV/microbe_cluster_matrix.csv")
# Save the metabolite cluster matrix as a CSV file
pd.DataFrame(data=metab_cluster_matrix, index=reduced_score_df.columns,
              columns=reduced_score_df.columns).to_csv("results/CV/metabolite_cluster_matrix.csv")

# Create DataFrame for metabolite clusters with an additional "Module" column
metabolite_cluster_df = pd.DataFrame(data=metabolite_clusters, index=binary_score_df.columns , columns=["Cluster"])
microbe_cluster_df = pd.DataFrame(data=microbe_clusters, index=binary_score_df.index, columns=["Cluster"])


# Initialize the "Module" column with 0 for metabolites
metabolite_cluster_df["Module"] = int(0)
for i, clu in enumerate(metabolite_cluster_df["Cluster"].unique()):
  metabolite_cluster_df.loc[metabolite_cluster_df["Cluster"] == clu, "Module"] = int(i)
# Sort the metabolite DataFrame by the "Module" column
metabolite_cluster_df = metabolite_cluster_df.sort_values(by="Module")

# Initialize the "Module" column with 0 for microbes
microbe_cluster_df["Module"] = int(0)
# Iterate over unique clusters and assign module numbers
for i, clu in enumerate(microbe_cluster_df["Cluster"].unique()):
  microbe_cluster_df.loc[microbe_cluster_df["Cluster"] == clu, "Module"] = int(i)
# Sort the microbe DataFrame by the "Module" column
microbe_cluster_df = microbe_cluster_df.sort_values(by="Module")

# Save both metabolite and microbe cluster DataFrames to CSV files
metabolite_cluster_df.to_csv("results/CV/metabolite_clusters.csv")
microbe_cluster_df.to_csv("results/CV/microbe_clusters.csv")

#Define full paths for saving the cluster files
metabolite_cluster_csv_path = os.path.join(folder_path, "metabolite_clusters.csv")
microbe_cluster_csv_path = os.path.join(folder_path, "microbe_clusters.csv")

# Save the DataFrames again to the specified folder paths
metabolite_cluster_df.to_csv(metabolite_cluster_csv_path)
microbe_cluster_df.to_csv(microbe_cluster_csv_path)



###################################################
# Bicluster Interaction Matrix_Actual Name
###################################################

binary_score_df_2 = binary_score_df.copy()

binary_score_df_2.index = binary_score_df_2.index.str.replace("SS_", "", regex=False)

# List of column names to search for in Metabolomic.csv
row_names_to_search = binary_score_df_2.columns 

# Find the column indices of the specified column names in Metabolomic.csv
column_indices = [metabolomic.columns.get_loc(name) for name in row_names_to_search]

# Find the corresponding column names in Metabolomic_H.csv
corresponding_column_names = [metabolomic_h.columns[i] for i in column_indices]
binary_score_df_2.columns=corresponding_column_names     

np.random.seed(42)
# Perform hierarchical clustering on both the microbiome and metabolite data
# Microbe tree: hierarchical clustering on the rows (microbes)
# Metabolite tree: hierarchical clustering on the columns (metabolites)
microbe_tree = shc.linkage(binary_score_df.values, method='complete')
metabolite_tree = shc.linkage(np.transpose(binary_score_df.values), method='complete')

# Cut the hierarchical tree to form clusters
# 'num_metabolite_clusters' and 'num_microbiome_clusters' represent the number of clusters for metabolites and microbes, respectively
metabolite_clusters = np.array(cut_tree(metabolite_tree, n_clusters=num_metabolite_clusters)).reshape(-1)
microbe_clusters = np.array(cut_tree(microbe_tree, n_clusters=num_microbiome_clusters)).reshape(-1)

metab_col_scale = 1/(num_metabolite_clusters-1)
micro_col_scale = 1/(num_microbiome_clusters-1)

wild_card_micro = np.random.uniform(0,1,num_microbiome_clusters)
wild_card_metab = np.random.uniform(0,1,num_metabolite_clusters)

# Generate a list of colors for each cluster, combining random and scaled components
micro_colors = [(wild_card_micro[x],1-x*micro_col_scale,x*micro_col_scale) for x in microbe_clusters]
# Metabolite colors: similar logic as for microbes, but the colors are different
metab_colors = [(1-x*metab_col_scale,wild_card_metab[x],x*metab_col_scale) for x in metabolite_clusters]

# Create a clustered heatmap using seaborn's clustermap
# binary_score_df is the input data (interaction score matrix)
sns.clustermap(binary_score_df_2, method="complete", row_colors = micro_colors, col_colors=metab_colors, row_linkage=microbe_tree,
                col_linkage=metabolite_tree, cmap = "coolwarm", figsize=(8,8), cbar_pos=(0.05, 0.88, 0.025, 0.10))

# Save the clustermap as a PNG file with high resolution
plt.savefig("results/clustermap.png", dpi=300)
# Save the clustermap as a PDF file with higher resolution
plt.savefig("results/clustermap3.pdf", dpi=600)

# Define the file path for saving the clustermap image in another directory
clustermap = os.path.join(folder_path, "Heatmap of Microbe-Metabolite Interactions_Actual_Name.png")
plt.savefig(clustermap)
clustermap = os.path.join(folder_path, "Heatmap of Microbe-Metabolite Interactions_Actual_Name.pdf")
plt.savefig(clustermap)


###################################################
# Determine Microbial Module Enrichment
###################################################

# Create masks to identify samples from G1 and G2 groups based on the 'Label' column
mask_con = diagnosis['Label'] == G1
mask_group_2 = diagnosis['Label']==G2

# Extract the sample indices (IDs) for G1 and G2 groups
g0 = diagnosis[mask_con].index
g1 = diagnosis[mask_group_2].index

#micro_sub = micro_comp_df
micro_sub = BB

enriched_in = []
p_list = []
micro_comp_cluster_df = pd.DataFrame(index=samples)
# Reinitialize the microbiome subset as a DataFrame, ensuring proper indexing
micro_sub = pd.DataFrame(index=micro_sub.index, columns=micro_sub.columns,
                         data = micro_sub)
microbiome_clusters=microbe_cluster_df

# Initialize dictionaries to store the names (samples) belonging to each microbiome cluster
names_with_true_dict = {}
names_with_true_dict2 = {}
    
# Loop over num_microbiome_clusters
for mc in range(num_microbiome_clusters):
    # Identify the samples that belong to the current cluster (mc)
    v = microbe_cluster_df["Cluster"] == mc
    names_with_true = v.index[v].tolist()  # Convert index to list
    names_with_true_dict[mc] = names_with_true  # Store names_with_true in the dictionary
    # Calculate the mean microbiome abundance for G1 and G2 groups within the current cluster
    g0_cluster = micro_sub.loc[g0, names_with_true].mean(1).values
    g1_cluster = micro_sub.loc[g1, names_with_true].mean(1).values
    # Add the mean microbiome module abundance for all samples to the main DataFrame
    micro_comp_cluster_df["Module " + str(mc)] = micro_sub.loc[:, names_with_true].mean(1).values

    # Perform Mann-Whitney U test to compare the microbiome module abundance between G1 and G2
    p_value = mannwhitneyu(g0_cluster, g1_cluster)[1]
    p_list.append(p_value)  #Store the p-value for later analysis

    # Check if the p-value is significant (less than 0.05), and determine which group is enriched
    if p_value < 0.05:
        p_value_one_sided = mannwhitneyu(g0_cluster, g1_cluster, alternative="greater")[1]
        if p_value_one_sided < 0.05:
            enriched_in.append(G1)
        else:
            enriched_in.append(G2)
    else:
        enriched_in.append("None")

font = {'family': 'normal',
        'weight': 'normal',
        'size': 10}
plt.rcParams['font.family'] = 'sans-serif'  # Replace 'sans-serif' with a font family available on your system.

# Create a DataFrame to store p-values and enriched groups for each microbiome module
micro_cluster_enrichment_df = pd.DataFrame(index=["Microbial Module " + str(x) for x in range(num_microbiome_clusters)])
micro_cluster_enrichment_df["p-value"] = p_list
micro_cluster_enrichment_df["Enriched"] = enriched_in
micro_comp_cluster_df["Diagnosis"] = diagnosis

# Save the enrichment results to a CSV file
micro_cluster_enrichment_df.to_csv("results/CV/microbiome_module_enrichment.csv")
microbiome_module_enrichment_csv_path = os.path.join(folder_path, "microbiome_module_enrichment.csv")
micro_cluster_enrichment_df.to_csv(microbiome_module_enrichment_csv_path)

micro_box_df = pd.melt(micro_comp_cluster_df, id_vars= ["Diagnosis"], value_vars=micro_comp_cluster_df.columns[0:num_microbiome_clusters])

# Create a boxplot to visualize the mean abundance of each microbiome module by diagnosis group
plt.figure(figsize=(8,8), dpi=300)
sns.boxplot(data=micro_box_df, x="variable", y="value", hue="Diagnosis")
plt.xlabel("Microbiome Module")
plt.ylabel("Mean Module Abundance")
plt.title("Microbiome Module by Label")
plt.savefig("results/micro_module_enrichment.png")



# Define a custom color palette for your boxplots
custom_palette = {
    G1: 'blue',
    G2: 'red'
}

# Modify the plotting code to add p-values for significant groups
plt.figure(figsize=(9, 9), dpi=300)
ax = sns.boxplot(data=micro_box_df, x="variable", y="value", hue="Diagnosis", palette=custom_palette)

plt.xlabel("Microbiome Module")
plt.ylabel("Mean Module Abundance")
plt.title("Microbiome Module by Label")

# Add p-values for significant groups
p_values = np.array(p_list)
n_modules = len(p_values)

# Get original x-axis tick positions
original_xtick_positions = np.arange(n_modules)

for i in range(n_modules):
    p_value = p_values[i]
    if p_value < 0.05:
        x_position = original_xtick_positions[i]
        y_position = np.min(micro_box_df["value"]) - 0.02  # Adjust this value for the desired vertical position
        if p_value <= 0.05:
            plt.text(x_position, y_position, f"p-value<{p_value:.3f}", fontsize=5, ha='center', va='center', color='red')
        else:
            plt.text(x_position, y_position, f"p-value<{abs(p_value-0.9*p_value):.3f}", fontsize=5, ha='center', va='center', color='red')
    
# Customize the x-axis tick labels without changing the position
ax.set_xticks(original_xtick_positions)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

# Customize the legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles[:2], labels[:2], title="Diagnosis", loc="upper left")

plt.tight_layout()  
plt.savefig("results/micro_module_enrichment.png")


micro_module_enrichment = os.path.join(folder_path, "Microbial module abundances enrichment.png")
plt.savefig(micro_module_enrichment)

micro_module_enrichment = os.path.join(folder_path, "Microbial module abundances enrichment.pdf")
plt.savefig(micro_module_enrichment)

plt.show()


metab_sub = AA

enriched_in = []
p_list = []
    
metab_comp_cluster_df = pd.DataFrame(index=samples)
metab_sub = pd.DataFrame(index=metab_sub.index, columns=metab_sub.columns,
                         data = metab_sub)
metab_sub = metab_sub[metabolite_cluster_df.index]

for mc in range(num_metabolite_clusters):
    v = metabolite_cluster_df["Cluster"] == mc
    names_with_true2 = v.index[v].tolist()  # Convert index to list
    names_with_true_dict2[mc] = names_with_true2  # Store names_with_true in the dictionary

    g0_cluster = metab_sub.loc[g0, names_with_true2].mean(1).values
    g1_cluster = metab_sub.loc[g1, names_with_true2].mean(1).values
    metab_comp_cluster_df["Module " + str(mc)] = metab_sub.loc[:, names_with_true2].mean(1).values
    p_value = mannwhitneyu(g0_cluster, g1_cluster)[1]
    p_list.append(p_value)

    if p_value < 0.05:
        p_value_one_sided = mannwhitneyu(g0_cluster, g1_cluster, alternative="greater")[1]
        if p_value_one_sided < 0.05:
            enriched_in.append(G1)
        else:
            enriched_in.append(G2)

    else:
        enriched_in.append("None")

p_values = np.array(p_list)
n_modules = len(p_values)

metab_cluster_enrichment_df = pd.DataFrame(index=["Metabolite Module " + str(x) for x in range(num_metabolite_clusters)])
metab_cluster_enrichment_df["p-value"] = p_list
metab_cluster_enrichment_df["Enriched"] = enriched_in
plt.rcParams['font.family'] = 'sans-serif'  # Replace 'sans-serif' with a font family available on your system.
metab_cluster_enrichment_df.to_csv("results/CV/metabolite_cluster_enrichment.csv")
metabolite_cluster_enrichment_csv_path = os.path.join(folder_path, "metabolite_module_enrichment.csv")
metab_cluster_enrichment_df.to_csv(metabolite_cluster_enrichment_csv_path)

metab_comp_cluster_df["Diagnosis"] = diagnosis




metab_box_df = pd.melt(metab_comp_cluster_df, id_vars= ["Diagnosis"], value_vars=metab_comp_cluster_df.columns[0:num_metabolite_clusters])
plt.figure(figsize=(8,8), dpi=300)
sns.boxplot(data=metab_box_df, x="variable", y="value", hue="Diagnosis")
plt.xlabel("Metabolite Module")
plt.ylabel("Mean Module Abundance")
plt.title("Metabolite Module by Label")
plt.savefig("results/metab_module_enrichment.png")


# Define a custom color palette for your boxplots
custom_palette = {
    G1: 'blue',
    G2: 'red'
}
# Modify the plotting code to add stars for significant groups
plt.figure(figsize=(8, 8), dpi=300)
sns.boxplot(data=metab_box_df, x="variable", y="value", hue="Diagnosis", palette=custom_palette)

plt.xlabel("Metabolite Module")
plt.ylabel("Mean Module Abundance")
plt.title("Metabolite Module by Label")

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right', fontsize=8)

for i in range(n_modules):
    p_value = p_values[i]
    if p_value < 0.05:
        x_position = i
        y_position = np.min(metab_box_df["value"]) - 0.02  # Adjust this value for the desired vertical position
        if p_value <= 0.05:
            plt.text(x_position, y_position, f"p-value<{p_value:.3f}", fontsize=5, ha='center', va='center', color='red')
        else:
           plt.text(x_position, y_position, f"p-value<{abs(p_value-0.9*p_value):.3f}", fontsize=5, ha='center', va='center', color='red')


# Customize the x-axis tick labels without changing the position
ax.set_xticks(original_xtick_positions)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

# Customize the legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles[:2], labels[:2], title="Diagnosis", loc="upper left")

plt.tight_layout()  
plt.savefig("results/metab_module_enrichment.png")
 

metab_module_enrichment = os.path.join(folder_path, "Metabolite module abundances enrichment.png")
plt.savefig(metab_module_enrichment)

metab_module_enrichment = os.path.join(folder_path, "Metabolite module abundances enrichment.pdf")
plt.savefig(metab_module_enrichment)
plt.show()

   
###################################################
# Network Connecting
###################################################
class NumpyFloat32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
network = {}
network["elements"] = {}
network["elements"]["nodes"] = []
network["elements"]["edges"] = []

# Initialize dictionaries to store most abundant names
most_abundant_microbiome = {}
most_abundant_metabolite = {}

for s in range(num_metabolite_clusters):
    node = {"data": {'id': "Metabolic_" + str(s)}}
    network["elements"]["nodes"].append(node)

for s in range(num_microbiome_clusters):
    node = {"data": {'id': "Microbiome_" + str(s)}}
    network["elements"]["nodes"].append(node)
score_list = []

for mic in range(num_microbiome_clusters):
    for met in range(num_metabolite_clusters):
        microbe_condition = (microbe_cluster_df['Cluster'] == mic)
        metabolite_condition = (metabolite_cluster_df['Cluster'] == met)

        filtered_binary_score_df = binary_score_df.loc[microbe_condition, metabolite_condition]

        A = reduced_score_df/sig_cutoff
        filtered_score_df = A.loc[microbe_condition, metabolite_condition]
        most_abundant_microbiome[(mic, met)] = filtered_binary_score_df.idxmax(axis=0).value_counts().idxmax()

        # Store all metabolite names
        most_abundant_metabolite[(mic, met)] = filtered_binary_score_df.idxmax(axis=1).value_counts().idxmax()

        score = np.round(np.mean(filtered_binary_score_df.values.flatten()), 3)


        edge = {
            "data": {
                "id": "Micro_" + str(mic) + "_to_Metab_" + str(met),
                "source": "Microbiome_" + str(mic),
                "target": "Metabolic_" + str(met),
                "score": score,
                "microbiome_name": most_abundant_microbiome[(mic, met)],
                "metabolite_name": most_abundant_metabolite[(mic, met)]
            }
        }
        network["elements"]["edges"].append(edge)
        score_list.append(score)

with open('./results/CV/network.json', 'w') as outfile:
    json.dump(network, outfile, cls=NumpyFloat32Encoder)

    


# List to store enriched modules for microbiome
enriched_modules_micro = []
enriched_in_G1_count_micro = 0
enriched_in_G2_count_micro = 0

# List to store enriched modules for metabolite
enriched_modules_metab = []
enriched_in_labels_metab = []
enriched_in_G1_count_metab = 0
enriched_in_G2_count_metab = 0

for mc in range(num_microbiome_clusters):
    v = microbe_cluster_df["Cluster"] == mc
    names_with_true = v.index[v == True]
    g0_cluster = micro_sub.loc[g0, names_with_true].mean(1).values
    g1_cluster = micro_sub.loc[g1, names_with_true].mean(1).values
    micro_comp_cluster_df["Module " + str(mc)] = micro_sub.loc[:, names_with_true].mean(1).values
    p_value = mannwhitneyu(g0_cluster, g1_cluster)[1]
    if p_value < 0.05:
        p_value_one_sided = mannwhitneyu(g0_cluster, g1_cluster, alternative="greater")[1]
        if p_value_one_sided < 0.05:
            enriched_modules_micro.append({"Module": "Microbiome_" + str(mc), "Enriched_in": G1})
            enriched_in_G1_count_micro += 1
        else:
            enriched_modules_micro.append({"Module": "Microbiome_" + str(mc), "Enriched_in": G2})
            enriched_in_G2_count_micro += 1
    else:
        enriched_modules_micro.append({"Module": "Microbiome_" + str(mc), "Enriched_in": "None"})

# Create a DataFrame for enriched modules (microbiome)
enriched_df_micro = pd.DataFrame(enriched_modules_micro)

    
# Number of modules enriched in CON and CIR (microbiome)

microbiome_nodes_reordered = ["Microbiome_" + str(x) for x in range(num_microbiome_clusters)]

# Update the order of microbiome nodes
enriched_G1_modules_MIC = enriched_df_micro[enriched_df_micro["Enriched_in"] == G1]["Module"].tolist()
enriched_G2_modules_MIC = enriched_df_micro[enriched_df_micro["Enriched_in"] == G2]["Module"].tolist()
non_enriched_modules_micro = [node for node in microbiome_nodes_reordered if node not in enriched_G1_modules_MIC + enriched_G2_modules_MIC]
microbiome_nodes_reordered = enriched_G1_modules_MIC + enriched_G2_modules_MIC + non_enriched_modules_micro


# List to store enriched modules for metabolite
enriched_modules_metab = []
enriched_in_labels_metab = []

for mc in range(num_metabolite_clusters):
    g0_cluster = metab_sub.loc[g0, metabolite_cluster_df["Cluster"]==mc].mean(1).values
    g1_cluster =  metab_sub.loc[g1, metabolite_cluster_df["Cluster"]==mc].mean(1).values
    metab_comp_cluster_df["Module " + str(mc)] = metab_sub.loc[:,metabolite_cluster_df["Cluster"]==mc].mean(1).values
    p_value = mannwhitneyu(g0_cluster, g1_cluster)[1]
    if p_value < 0.05:
        p_value_one_sided = mannwhitneyu(g0_cluster, g1_cluster, alternative="greater")[1]
        if p_value_one_sided < 0.05:
            enriched_modules_metab.append({"Module": "Metabolic_" + str(mc), "Enriched_in": G1})
            enriched_in_G2_count_metab += 1
        else:
            enriched_modules_metab.append({"Module": "Metabolic_" + str(mc), "Enriched_in": G2})
            enriched_in_G1_count_metab += 1
    else:
        enriched_modules_metab.append({"Module": "Metabolic_" + str(mc), "Enriched_in": "None"})

    # Create a DataFrame for enriched modules (metabolite)
    enriched_df_metab = pd.DataFrame(enriched_modules_metab)
    

metabolite_nodes_reordered = ["Metabolic_" + str(x) for x in range(num_metabolite_clusters)]

# Update the order of metabolite nodes
enriched_in_labels_metab = enriched_df_metab[enriched_df_metab["Enriched_in"] != "None"]["Enriched_in"].tolist()
enriched_G2_modules_METAB = enriched_df_metab[enriched_df_metab["Enriched_in"] == G1]["Module"].tolist()
enriched_G1_modules_METAB = enriched_df_metab[enriched_df_metab["Enriched_in"] == G2]["Module"].tolist()
non_enriched_modules_metab = [node for node in metabolite_nodes_reordered if node not in enriched_G1_modules_METAB + enriched_G2_modules_METAB]
metabolite_nodes_reordered = enriched_G1_modules_METAB + enriched_G2_modules_METAB + non_enriched_modules_metab


# Create a new variable based on "Enriched_in"
enriched_df_micro['Enrichment_Category'] = enriched_df_micro['Enriched_in'].map({G1: G1, G2: G2}).fillna('None')
# Sort modules based on the new variable
df_sorted_MIC = enriched_df_micro.sort_values(by='Enrichment_Category')
microbiome_nodes_reordered = df_sorted_MIC['Module'].tolist()


# Create a new variable based on "Enriched_in"
enriched_df_metab['Enrichment_Category'] = enriched_df_metab['Enriched_in'].map({G1: G1, G2: G2}).fillna('None')
# Sort modules based on the new variable
df_sorted_MET = enriched_df_metab.sort_values(by='Enrichment_Category')
metabolite_nodes_reordered = df_sorted_MET['Module'].tolist()



########################################################
# Network connecting microbial with metabolomic modules
########################################################

# Load your network data from the JSON file
with open('./results/CV/network.json', 'r') as infile:
    network_data = json.load(infile)
all_metabolite_names = []
    
# Create an empty bipartite graph
G = nx.Graph()
fig, ax = plt.subplots(figsize=(18, 12))  # Increase the figure size

# Define color lists
metab_color_list = ["red", "blue", "orange", "yellow", "green", "tan", "gray", "turquoise", "lightsteelblue", "chocolate", "khaki", "lime"]
microbe_color_list = ["purple", "teal", "lightpink", "lightblue", "brown", "magenta", "lemonchiffon", "navy", "palegreen", "darkorange", "salmon", "black", "lightgreen", "red", "pink", "skyblue", "cyan", "teal", "lightyellow", "olive"]

for i, node_id in enumerate(metabolite_nodes_reordered):
    cluster_color = metab_color_list[i % len(metab_color_list)]

    # Define distances between groups and within group
    distance_between_groups = 7.0
    distance_within_group = 1.0
    if i < enriched_in_G2_count_metab:
        G.add_node(node_id, bipartite=0, pos=(1, i * distance_within_group), color=cluster_color)
    elif enriched_in_G2_count_metab <= i < enriched_in_G1_count_metab + enriched_in_G2_count_metab:
        G.add_node(node_id, bipartite=0, pos=(1, distance_between_groups + (i - 2) * distance_within_group), color=cluster_color)
    else:
        G.add_node(node_id, bipartite=0, pos=(1, 2 * distance_between_groups + (i - 6) * distance_within_group), color=cluster_color)

# Add microbial nodes with colors based on cluster
for i, node_id in enumerate(microbiome_nodes_reordered):
    cluster_color = microbe_color_list[i % len(microbe_color_list)]

    # Define distances between groups and within group
    distance_between_groups = 7.0
    distance_within_group = 1.0
    if i < enriched_in_G2_count_micro:
        G.add_node(node_id, bipartite=1, pos=(-1, i * distance_within_group), color=cluster_color)
    elif enriched_in_G2_count_micro <= i < enriched_in_G1_count_micro + enriched_in_G2_count_micro:
        G.add_node(node_id, bipartite=1, pos=(-1, distance_between_groups + (i - 3) * distance_within_group), color=cluster_color)
    else:
        G.add_node(node_id, bipartite=1, pos=(-1, 2 * distance_between_groups + (i - 6) * distance_within_group), color=cluster_color)
    
# Add edges to the graph with scores > 0.25
for edge_data in network_data['elements']['edges']:
    source = edge_data['data']['source']
    target = edge_data['data']['target']
    score = edge_data['data']['score']
    microbiome_name = edge_data['data']['microbiome_name']
    metabolite_name = edge_data['data']['metabolite_name']

    if abs(score) > 0.25:
        color = 'red' if score > 0 else 'blue'
        G.add_edge(source, target, weight=score, color=color, microbiome_name=microbiome_name, metabolite_name=metabolite_name)

# Get positions from node attributes
pos = nx.get_node_attributes(G, 'pos')

# Draw nodes with colors
metabolite_nodes = [node for node in G.nodes if node.startswith("Metabolic_")]
microbiome_nodes = [node for node in G.nodes if node.startswith("Microbiome_")]

# Draw metabolite nodes with colors and custom positions
nx.draw_networkx_nodes(G, pos=pos, nodelist=metabolite_nodes, node_color=[G.nodes[node]['color'] for node in metabolite_nodes], node_size=1000)
# Draw microbial nodes with colors and custom positions
nx.draw_networkx_nodes(G, pos=pos, nodelist=microbiome_nodes, node_color=[G.nodes[node]['color'] for node in microbiome_nodes], node_size=1000)

# Draw edges with colors
nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(data=True), width=2, edge_color=[G.edges[edge]['color'] for edge in G.edges])
    
# Draw labels with adjusted positions and microbiome_name/metabolite_name
labels = {}
for node in G.nodes:
    node_name = node.split('_')[1]  # Extract the cluster number

    # Add microbiome_name to microbiome nodes
    if node.startswith("Microbiome_"):
        relevant_edges = [G.edges[(_, node)] for _ in G.neighbors(node) if abs(G.edges[(_, node)]['weight']) > 0.25]
        unique_names = set(edge['microbiome_name'] for edge in relevant_edges)
        # Remove the prefix 'SS_' from each name in unique_names
        unique_names = {name.replace("SS_", "") for name in unique_names}
        labels[node] = f"{', '.join(unique_names)}"

        # Add metabolite_name to metabolite nodes
    if node.startswith("Metabolic_"):
        relevant_edges = [G.edges[(_, node)] for _ in G.neighbors(node) if abs(G.edges[(_, node)]['weight']) > 0.25]
        metabolite_names = [edge['metabolite_name'] for edge in relevant_edges]
        unique_names = set(metabolite_names)
        # List of column names to search for in Metabolomic.csv
        column_names_to_search = unique_names
        # Find the column indices of the specified column names in Metabolomic.csv
        column_indices = [metabolomic.columns.get_loc(name) for name in column_names_to_search]
        # Find the corresponding column names in Metabolomic_H.csv
        unique_names= corresponding_column_names = [metabolomic_h.columns[i] for i in column_indices]
        # Remove the prefix 'MET_' from each name in unique_names
        unique_names = {name.replace("MET_", "") for name in unique_names}
        labels[node] = f"{', '.join(unique_names)}"
        all_metabolite_names.extend(unique_names)

all_metabolite_names = list(set(all_metabolite_names))

# Draw labels with adjusted positions
for node, (x, y) in pos.items():
    num = ''.join(filter(str.isdigit, node))  # Extract numeric part
    if node.startswith("Microbiome_"):
        plt.text(x - 0.2, y, labels[node], fontsize=12, ha='right', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
        plt.text(x, y, num, fontsize=14, ha='center', va='center')
    elif node.startswith("Metabolic_"):
        plt.text(x + 0.2, y, labels[node], fontsize=12, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
        plt.text(x, y, num, fontsize=14, ha='center', va='center')


# # Draw text labels for "Microbial Module" at the top left
# plt.text(-1,14, "Microbial Module", ha="center", fontsize=16)

# # Draw text labels for "Metabolite Module" at the top right
# plt.text(1, 14, "Metabolite Module", ha="center", fontsize=16)
    
# Find the maximum Y values for Metabolite and Cytokine nodes
Microbial_max_y = max(y for node, (x, y) in pos.items() if "Microbiome_" in node)
Metabolite_max_y = max(y for node, (x, y) in pos.items() if "Metabolic_" in node)

# Add titles at the top, just above the highest node in each side
plt.text(-1, Microbial_max_y + 1, "Microbial Module", ha="center", fontsize=16, fontweight='bold')
plt.text(1, Metabolite_max_y + 1, "Metabolite Module", ha="center", fontsize=16, fontweight='bold')


relashiship = os.path.join(folder_path, "Network connecting microbial with metabolomic modules.png")
plt.savefig(relashiship, dpi=300, bbox_inches='tight')  # Use dpi to set resolution and bbox_inches to ensure no part is cut off

relashiship = os.path.join(folder_path, "Network connecting microbial with metabolomic modules.pdf")
plt.savefig(relashiship, dpi=300, bbox_inches='tight')  # Use dpi to set resolution and bbox_inches to ensure no part is cut off
plt.show()
