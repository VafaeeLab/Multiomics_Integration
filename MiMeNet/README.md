# Exploring Microbiome, Metabolite, and Cytokine Interactions in MAFLD Cirrhosis: A MiMeNet-Driven Multi-Omics Integration

This repository contains the code and data for the study **"Exploring Microbiome, Metabolite, and Cytokine Interactions in MAFLD Cirrhosis: A MiMeNet-Driven Multi-Omics Integration"**.

---

## **Overview**

This project utilizes the MiMeNet (Microbiome-Metabolome Network) framework to investigate the complex interactions between the gut microbiome, metabolome, and cytokines in metabolic-associated fatty liver disease (MAFLD) related cirrhosis. The analysis integrates microbiome abundance data, metabolite profiles, and cytokine measurements from:
- **28 MAFLD-cirrhosis patients**
- **28 matched healthy controls**

---

## **Requirements**

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- TensorFlow
- KNNImputer

---

## **Installation**
Clone this repository and install the required packages:

```bash
git clone [https://github.com/Karimi-M/MiMeNet-Analysis.git](https://github.com/Karimi-M/MiMeNet-Analysis.git)
cd MiMeNet-Analysis
pip install -r requirements.txt
```

---

## **This script performs the following steps**

 - Data Preprocessing and Normalization
 - MiMeNet Model Training and Evaluation 
 - Identification of Well-Predicted Metabolites
 - Generation of Attribution Score Matrix
 - Biclustering Analysis
 - Module-Based Interaction Network Construction
 - Integration of Cytokine Data
 - Statistical Analysis and Visualization

---
## **Data Preparation**

The script expects the following input files:

- **Microbiome data**: `Stool_Species.csv`
- **Metabolite data**: `Metabolomic.csv`
- **Metabolite metadata**: `MET_Annotation.csv`
- **Diagnosis information**: `diagnosis.csv`
- **Cytokine data**: `Cytokine.csv`


---

## **Usage**
The main analysis script performs the following steps:

1. **Data loading and preprocessing**
2. **Group definition and sample selection**
3. **Feature filtering based on zero values and missing data**
4. **Data normalization** 
   - CPM and log transformation for microbiome data
   - Scale normalization for metabolite data
5. **Imputation of missing values using KNN**
6. **Feature selection** based on differential expression analysis and fold change
7. **MiMeNet model training and cross-validation**
8. **Result visualization and analysis**


---

## **Key Functions**  

```python
def scale_normalization(df):
    # Perform scale normalization on metabolite data
    ...

def cpm_and_log_transformation(df):
    # Perform CPM and log transformation on microbiome data
    ...

def differential_expression(data, labels, p_value_threshold=0.15):
    # Perform differential expression analysis
    ...

def fold_change_filter(data, labels, fold_change_threshold=1.5):
    # Filter features based on fold change

```
---
## **Explanation**
- **Key Functions**: The content within triple backticks (```) is recognized as a code block, which users can copy.
- **Next Section Title**: This is just regular Markdown text, and will not allow for easy copying of content directly (e.g., plain text, descriptions, etc.).

**Important**: The `<br>` tag is used for a line break if you need a little extra space between sections for visual clarity, but it is optional. It helps if your section transitions look cramped.


---

## **Cross-Validation and Model Training**

The script performs multiple runs of k-fold cross-validation:

- **Number of runs:** 10
- **Number of folds:** 10

For each fold:
- Data is split into training and test sets
- Data is log-transformed and scaled
- MiMeNet model is trained on the training set
- Predictions are made on the test set
- Results are saved (predictions and score matrices)

---




## Results
The analysis results are saved in timestamped directories within the specified output folder. Each run and fold has its own subdirectory containing:
 - Training and test set data
 - Predictions
 - Score matrices
Additionally, the script generates:
 - Correlation matrices for cross-validated evaluation
 - Visualizations of the results
---



## Customization

 **Input Data and Group Definitions**

 - Input file paths:
	 - Microbiome data: Input_path
	 - Metabolite data: Output_path
	 - Metabolite metadata: "MET_Annotation.csv"
	 - Diagnosis information: "diagnosis.csv"
 - Group definitions
	 - G1, G2, G3, G4, G5: Define different groups (e.g., "CON", "CIR",
	   "LN", "LX", "LN+LX=HCC")
	 - classes_to_extract: Specify which groups to include in the analysis
   
**Data Preprocessing**
**Filtering thresholds:**
 - Zero value threshold: threshold = 0.9 * micro_df.shape Missing value
 - threshold: threshold = 50

**Normalization methods:**
 - scale_normalization(): For metabolite data
 - cpm_and_log_transformation(): For microbiome data

**Imputation method:**
 - KNN Imputation: KNNImputer(n_neighbors=5)
 - MICEImputation
   
**Feature Dimenssion Reduction:**
 - Differential expression analysis:
	 - P-value threshold: p_value_threshold=0.15
 - Fold change filtering:
	 - Fold change threshold: fold_change_threshold=1.5
  - 
**Cross-Validation**
 - Number of runs: num_run_cv = 10
 - Number of folds: num_cv = 10

**Neural Network Architecture**

The neural network architecture in this script is highly customizable through the MiMeNet class. Here are the key parameters you can adjust:
- **num_layer:** Number of hidden layers in the network
- **layer_nodes:** List specifying the number of nodes in each hidden layer
- **l1:** L1 regularization parameter
- **l2:** L2 regularization parameter
- **dropout:** Dropout rate for regularization
- **learning_rate:** Learning rate for the optimizer

These parameters are loaded from a JSON file ("network_parameters.txt"), or they can be set manually in the script. You can modify these parameters to experiment with different network architectures:

---

## Troubleshooting

If you encounter issues with loading network parameters, the script will output a warning and proceed with default values.

---

## Contact

For any questions or issues, please open an issue on this GitHub repository or contact [m.karimim@unsw.edu.au].
