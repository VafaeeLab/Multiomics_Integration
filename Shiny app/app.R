# Disable warnings
options(warn = -1)

# Load required libraries
library(shiny)
library(shinydashboard)
library(shinymaterial)
library(shinyWidgets)
library(BiocManager)
library(caret)
library(dplyr)
library(readr)
library(DT)
library(ranger)
library(VIM)
library(lefser)
library(Boruta)
library(SummarizedExperiment)
library(e1071)
library(xgboost)
library(pROC)
library(tidyverse)
library(ggVennDiagram)
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(dismo)
library(rlang)
library(Rtsne)
library(umap)
library(vegan)
library(reactable)
library(profvis)
library(sva)
library(DMwR)
library(gbm)
library(mice)
library(reshape2)

# Define the UI
ui <- dashboardPage(
  dashboardHeader(
    title = "Liver Disorders:MRFF",
    tags$li(class = "dropdown",
            tags$img(src = "UNSW_Logo.png", height = 50, width = 50, align = "right"))
  ),
  
  dashboardSidebar(
    sidebarMenu(
      # Preprocessing tab
      menuItem(
        "Preprocessing",
        tabName = "preprocessing",
        icon = icon("cogs")
      ),
      
      # Feature Selection tab
      menuItem(
        "Feature Selection",
        tabName = "feature_selection",
        icon = icon("chart-bar")
      ),
      
      # Classification tab
      menuItem(
        "Classification",
        tabName = "classification",
        icon = icon("list-alt")
      ),
      
      # External Validation tab
      menuItem(
        "External Validation",
        tabName = "external_validation",
        icon = icon("external-link-alt")
      )
    )
  ),
  
  dashboardBody(
    tabItems(
      # Preprocessing tab
      tabItem(
        tabName = "preprocessing",
        fluidPage(
          sidebarLayout(
            sidebarPanel(
              # Preprocessing options
              h3("Preprocessing", style = "font-size: 12px;"),
              radioButtons("classificationType", "Classification Type", choices = c("Binary", "Multiclass"), inline = TRUE, selected = "Binary"),
              
              # Group selection
              fluidRow(
                column(
                  width = 6,
                  checkboxGroupInput(
                    "leftLabelSelection",
                    HTML("<sup>1<sup>st</sup></sup> Group"),
                    choices = c("CIR", "CON", "LN", "LX","HCC=[LN+LX]"),
                    selected = "CON"
                  )
                ),
                column(
                  width = 6,
                  conditionalPanel(
                    condition = "input.classificationType == 'Binary'",
                    checkboxGroupInput(
                      "rightLabelSelection",
                      HTML("<sup>2<sup>nd</sup></sup> Group"),
                      choices =c("CIR", "CON", "LN", "LX","HCC=[LN+LX]"),
                    )
                  )
                )
              ),
              
              
              # Remove zero values
              sliderInput(
                "removeZeroSlider",
                "Remove Zero Values (%)",
                min = 0, max = 100, value = 90, step = 5
              ),
              actionButton("removeZeroButton", "Remove Zero Values"),
              
              # Remove missing values
              sliderInput(
                "removeMissingSlider",
                "Remove Missing Values (%)",
                min = 0, max = 100, value = 50, step = 5
              ),
              actionButton("removeMissingButton", "Remove Missing Values"),
              # Normalization method
              selectInput(
                "normalizationMethod", "Normalization Method",
                choices = c("None","Scale", "CPM", "CPM+Log", "Min-Max"),
                selected = "None"
              ),
              actionButton("normalizeButton", "Normalize Datasets"),
              # Imputation method
              selectInput(
                "imputationMethod", "Imputation Method",
                choices = c("None", "Mean", "kNN", "MICE"),
                selected = "None"
              ),
              conditionalPanel(
                condition = "input.imputationMethod == 'kNN'",
                numericInput("kValue", "k Value for kNN", value = 5, min = 1, step = 1)
              ),
              actionButton("imputeButton", "Impute Datasets"),
              
              
            ),
            
            mainPanel(
              tabsetPanel(
                # Upload Dataset tab
                tabPanel(
                  "Upload Dataset",
                  textOutput("processText"),
                  div(
                    id = "resetDatasetContainer",
                    class = "pull-right",
                    actionButton("resetBtn", "Reset")
                  ),
                  checkboxGroupInput(
                    "datasetCheckboxes", "Select Dataset",
                    choices = c(
                      "Stool Species", "Stool Genus", "Oral Species", "Oral Genus", "Pathologic", "Cytokine", "Clinical", "Metabolomic","Lipoprotein","Small Molecule"
                    )
                  )
                ),
                
                # Data Table tab
                tabPanel(
                  "Data Table",
                  DT::dataTableOutput("selectedDatasetTable")
                ),
                
                # Data Description tab
                tabPanel(
                  "Data Description",
                  textOutput("datasetname"),
                  textOutput("dimText"),
                  fluidRow(
                    column(
                      width = 6,
                      sliderInput(
                        "obs_slider",
                        "Observation",
                        min = 2,
                        max = 107,
                        value = 2,
                        step = 1,
                        animate = TRUE
                      )
                    ),
                    column(
                      width = 6,
                      verbatimTextOutput("summary_output"),
                      plotOutput("stat_plot")  # Add plotOutput element to display the plot
                    )
                  )
                ),
                
                # Dataset Visualization tab
                tabPanel(
                  "Dataset Visualization",
                  selectInput("methodSelect", "Select Method", choices = c("t-SNE", "UMAP", "PCA", "Diversity")),
                  conditionalPanel(
                    condition = "input.methodSelect == 'Diversity'",
                    checkboxInput("simpsonCheckbox", "Simpson Diversity"),
                    checkboxInput("shannonCheckbox", "Shannon Diversity")
                  ),
                  plotOutput("visualizationPlot"),
                  downloadButton("DownloadPlot", "Download Plot")
                )
              )
            )
          )
        )
      ),
      
      # Feature Selection tab
      tabItem(
        tabName = "feature_selection",
        fluidPage(
          sidebarLayout(
            sidebarPanel(
              # Feature selection options
              h3("Feature Selection", style = "font-size: 12px;"),
              selectInput(
                "featureSelectionMethod",
                "Feature Selection Method",
                choices = c("Boruta", "Ranger", "Lefse", "Wilcoxon","Random")
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Ranger'",
                checkboxInput("impurity", "Mean Decrease Gini"),
                checkboxInput("permutation", "Mean Decrease Accuracy"),
                conditionalPanel(
                  condition = "input.featureSelectionMethod == 'Ranger'",
                  numericInput("numIterations", "Number of Iterations:", value = 50, min = 1),
                  conditionalPanel(
                    condition = "input.featureSelectionMethod == 'Ranger'",
                    numericInput("numTOP", "Number of TOP Features:", value = 10, min = 1),
                    conditionalPanel(
                      condition = "input.featureSelectionMethod == 'Ranger'",
                      sliderInput("percentageTOP", "High-Frequency Features (%)", min = 0, max = 100, value = 20,step = 5)
                    )
                  )
                )
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Boruta'",
                numericInput("numIterationsB", "Number of Iterations:", value = 50, min = 1)
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Boruta'",
                sliderInput("percentageTOPB", "High-Frequency Features (%)", min = 0, max = 100, value = 70,step = 5)
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Wilcoxon'",
                numericInput("numGroups", "Number of TOP Features:", value = 10, min = 1)
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Random'",
                numericInput("numfeatures", "Number of TOP Features:", value = 5, min = 1)
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Lefse'",
                numericInput("kruskalthreshold", "kruskal threshold:", value = 0.05, min = 0)
              ),
              conditionalPanel(
                condition = "input.featureSelectionMethod == 'Lefse'",
                numericInput("wilcoxthreshold", "wilcox threshold:", value = 0.05, min = 0)
              ),
              actionButton("featureSelectionButton", "Perform Feature Selection")
            ),
            mainPanel(
              tabsetPanel(
                
                tabPanel("Features Displaying", 
                         plotOutput("Featureplot"),
                         conditionalPanel(
                           condition = 'input$selectedTab == "Features Displaying"',
                           downloadButton("plotsave", "Download Plot")
                         )
                ),
                tabPanel("Selected Features Name", textOutput("selectedFeaturesOutput")),
                tabPanel("Relative Abundances Displaying", 
                         plotOutput("Abundancesplot"),
                         conditionalPanel(
                           condition = 'input$selectedTab == "Relative Abundances Displaying"',
                           downloadButton("plotsave2", "Download Plot")
                         )
                ),
                
              )
            )
          )
        )
      ),
      
      # Classification tab
      tabItem(
        tabName = "classification",
        fluidPage(
          sidebarLayout(
            sidebarPanel(
              # Classification options
              selectInput("SplittingMethod", "Splitting Method", choices = c("K-Fold", "Train-Test_Split", "LOOCV")),
              conditionalPanel(
                condition = "input.SplittingMethod== 'K-Fold'",
                numericInput("numIterationsC", "Number of Iterations:", value = 50, min = 1)
              ),
              conditionalPanel(
                condition = "input.SplittingMethod == 'K-Fold'",
                numericInput("kInput", "Number of Folds (k)", value = 5, min = 2),
              ),
              conditionalPanel(
                condition = "input.SplittingMethod == 'Train-Test_Split'",
                sliderInput("trainPercentageInput", "Train Percentage", value = 80, min = 0, max = 100,step = 5),
              ),
              selectInput("classificationMethod", "Classification Method", choices = c("Random Forest","GBM","XGBoost", "SVM")),
              actionButton("classificationButton", "Perform Classification")
            ),
            mainPanel(
              h4("Classification Results"),
              textOutput("sensitivityOutput"),
              textOutput("accuracyOutput"),
              textOutput("specificityOutput"),
              textOutput("f1scoreOutput"),
              textOutput("aucOutput"),
              textOutput("allCombinedNames"),
              textOutput("iteration"),
              plotOutput("boxplotOutput"),
              plotOutput("additionalPlotOutput"),
              downloadButton("downloadPlot", "Download Plot")
            )
          )
        )
      ),
      
      # External Validation tab
      tabItem(
        tabName = "external_validation",
        fluidPage(
          sidebarLayout(
            sidebarPanel(
              # External validation options
              h3("Ven Diagram", style = "font-size: 16px;"),
              actionButton("VenDiagramButton", "Perform Ven Diagram"),
              selectInput("datasetOption", "Select Dataset Option", choices = c("Loomba et al.")),
              selectInput(
                "BatchMethod", "Batch Effect Correction",
                choices = c("None", "ComBat"),
                selected = "None"
              ),
              checkboxInput("oversampleCheckbox", "Perform Oversampling"),
              actionButton("BatchButton", "Batch Correction"),
              selectInput("classificationMethod_v", "Classification Method", choices = c("Trained Model")),
              actionButton("classificationButton_v", "Perform Classification")
            ),
            mainPanel(
              tabsetPanel(
                # Model Evaluation tab
                tabPanel("Model Evaluation",
                         br(),
                         uiOutput("datasetNameOutput"),  # Output to display dataset name
                         br(),
                         uiOutput("datasetCheckboxes_EV"),
                         br(),
                         plotOutput("ModelEvaluation")
                ),
                
                # Ven Diagram tab
                tabPanel("Veen Diagram",
                         br(),
                         plotOutput("VenDiagram", height = "500px", width = "500px"),
                         downloadButton("downloadVenDiagram", "Download Venn Diagram")
                )
              ),
              
              h4("Classification Results"),
              textOutput("sensitivityOutput_v"),
              textOutput("accuracyOutput_v"),
              textOutput("specificityOutput_v"),
              textOutput("f1scoreOutput_v"),
              textOutput("aucOutput_v"),
              textOutput("allCombinedNames_v"),
              plotOutput("boxplotOutput_v"),
              downloadButton("downloadPlot_v", "Download Plot")
            )
          )
        )
      )
    )
  )
)


# Server code

server <- function(input, output, session) {
  # Reactive values
  datasetList <- reactiveVal(list())                # Original data set
  cleanedDatasetList <- reactiveVal(list())         # Cleaned data set
  selectedFeaturesOutput <- reactiveVal(NULL)       # Selected features
  DataDescription <- reactiveVal(NULL)              # Data description
  selectedDatasets <- reactiveVal(NULL)             # Selected data set for more analysis
  selectedDataset <- reactiveVal(NULL)              # Currently selected data set
  processText <- reactiveVal(NULL)                  # Text for process updates
  featureSelectionResults <- reactiveVal(NULL)      # Results of Selected features
  classificationResults <- reactiveVal(NULL)        # Results of classification method
  classificationModel <- reactiveVal(NULL)          # Trained classification model
  rangerselectedfeatures <- reactiveVal(NULL)       # Selected features by Ranger method
  lefseselectedfeatures <- reactiveVal(NULL)        # Selected features by Lefse method
  wilcoxonselectedfeatures <- reactiveVal(NULL)     # Selected features by Wilcoxon method
  broutaselectedfeatures <- reactiveVal(NULL)       # Selected features by Brouta method
  combinedName <- reactiveVal("")                   # Combined name of selected data set
  allCombinedNames <- reactiveVal("")               # Combined names of all selected data set
  fileNames <- reactiveVal(NULL)                    # File names of selected data set
  selectedDatasetsList <- reactiveVal(list())       # List of selected data set
  dataset <- reactiveVal(list())                    # Currently selected data set
  datasets_EV <- reactiveVal(list())                # External validation data set
  datasets_ORG <- reactiveVal(list())               # Original data set
  S <- reactiveVal(list())                          # List of data set
  datasets <- reactiveVal(list())                   # Combined list of data set
  generatedPlot <- reactiveValues(plot = NULL)      # Generated plot
  Plot <- reactiveVal(NULL)                         # Plot data
  selectedDatasets_V <- reactiveVal(NULL)           # Selected data set for validation
  
  # Define the dataset files and their names
  datasetFiles <- list(
    Dataset1 = "Stool Species.csv",
    Dataset2 = "Stool Genus.csv",
    Dataset3 = "Oral Species.csv",
    Dataset4 = "Oral Genus.csv",
    Dataset5 = "Pathologic.csv",
    Dataset6 = "Cytokine.csv",
    Dataset7 = "Clinical.csv",
    Dataset8 = "Metabolomic.csv",
    Dataset9 = "Lipoprotein.csv",
    Dataset10 = "small molecule.csv"
  )
  
  datasetFileNames <- names(datasetFiles)
  
  # Event handler for dataset checkboxes
  observeEvent(input$datasetCheckboxes, {
    selectedCheckboxes <- input$datasetCheckboxes
    selectedIndices <- sapply(selectedCheckboxes, function(checkbox) {
      switch(checkbox,
             "Stool Species" = 1,
             "Stool Genus" = 2,
             "Oral Species" = 3,
             "Oral Genus" = 4,
             "Pathologic" = 5,
             "Cytokine" = 6,
             "Clinical" = 7,
             "Metabolomic" = 8,
             "Lipoprotein"=9,
             "Small Molecule"=10
      )
    })
    selectedFiles <- datasetFiles[selectedIndices]
    
    # Read the selected datasets
    datasets_ORG <- list(
      Dataset1 = read.csv("Stool Species.csv"),
      Dataset2 = read.csv("Stool Genus.csv"),
      Dataset3 = read.csv("Oral Species.csv"),
      Dataset4 = read.csv("Oral Genus.csv"),
      Dataset5 = read.csv("Pathologic.csv"),
      Dataset6 = read.csv("Cytokine.csv"),
      Dataset7 = read.csv("Clinical.csv"),
      Dataset8 = read.csv("Metabolomic.csv"),
      Dataset9 =  read.csv("Lipoprotein.csv"),
      Dataset10 =  read.csv("small molecule.csv")
    )
    
    # Extract the file names
    fileNames <- sapply(datasetFileNames[selectedIndices], function(name) {
      # Remove the file extension (assuming all files are CSV)
      gsub(".csv$", "", name)
    })
    
    # Filter the datasets based on the selected checkboxes
    datasets_ORG <- datasets_ORG[fileNames]
    datasets_ORG(datasets_ORG)
    
    # Combine the datasets based on the checkboxes
    if (!is.null(input$datasetCheckboxes_EV) && !is.null(input$datasetCheckboxes)) {
      datasets <- c(datasets_EV(), datasets_ORG)
    } else {
      datasets <- datasets_ORG
    }
    datasets(datasets)
    
    # Combine the file names with an underscore
    combinedNames <- paste(selectedCheckboxes, collapse = "_")
    combinedName(combinedNames)                       # Update the value of combinedName
    fileNames(fileNames)
    
    # Update the output text
    output$combinedName <- renderText({
      paste("Classification Result:", combinedName())
    })
    
    datasetList(datasets)
    S <- datasetList()
    unique_names <- names(S)[!duplicated(names(S))]
    S <- S[unique_names]
    
  })
  
  # Define the dataset files for external validation
  datasetFiles_EV <- list(
    Dataset1_EV = "Stool Species_External.csv",
    Dataset2_EV = "Stool Genus_External.csv",
    Dataset3_EV = "Stool Species-External.csv",
    Dataset4_EV = "Stool Genus-External.csv"
  )
  
  datasetFileNames_EV <- names(datasetFiles_EV)
  
  # Event handler for external validation dataset checkboxes
  observeEvent(input$datasetCheckboxes_EV, {
    selectedCheckboxes_EV <- input$datasetCheckboxes_EV
    selectedIndices_EV <- sapply(selectedCheckboxes_EV, function(checkbox) {
      switch(checkbox,
             "Stool Species_External" = 1,
             "Stool Genus_External" = 2,
             "Stool Species-External" = 3,
             "Stool Genus-External" = 4
      )
    })
    selectedFiles_EV <- datasetFiles_EV[selectedIndices_EV]
    
    # Read the selected external validation datasets
    datasets_EV <- list(
      Dataset1_EV = read.csv("Stool Species_External.csv"),
      Dataset2_EV = read.csv("Stool Genus_External.csv"),
      Dataset3_EV = read.csv("Stool Species-External.csv"),
      Dataset4_EV = read.csv("Stool Genus-External.csv")
    )
    
    # Extract the file names
    fileNames <- sapply(datasetFileNames_EV[selectedIndices_EV], function(name) {
      # Remove the file extension (assuming all files are CSV)
      gsub(".csv$", "", name)
    })
    
    datasets_EV <- datasets_EV[fileNames]
    datasets_EV(datasets_EV)
    
    # Combine the external validation datasets with the original datasets
    if (!is.null(input$datasetCheckboxes) && !is.null(input$datasetCheckboxes_EV)) {
      # Check if the selected datasets have already been joined
      if (!identical(datasets(), datasets_EV)) {
        datasets <- c(datasets(), datasets_EV)
      }
    } else {
      datasets <- datasets_EV
    }
    datasets(datasets)
    
    # Combine the file names with an underscore
    combinedNames <- paste(selectedCheckboxes_EV, collapse = "_")
    combinedName(combinedNames)  # Update the value of combinedName
    fileNames(fileNames)
    output$combinedName <- renderText({
      paste("Classification Result:", combinedName())
    })
    
    datasetList(datasets)
    S <- datasetList()
    unique_names <- names(S)[!duplicated(names(S))]
    S <- S[unique_names]
    datasetList(S)
    
  })
  
  
  output$datasetCheckboxes_EV <- renderUI({
    option <- input$datasetOption
    datasetName <- if (option == "Loomba et al.") "Loomba et al." else "Leung et al."
    output$datasetNameOutput <- renderText({
      if (datasetName == "Loomba et al.") {
        paste("Dataset Name:", datasetName, shiny::a("Read Paper", href = "https://www.sciencedirect.com/science/article/pii/S1550413117302061"))
      } else if (datasetName == "Leung et al.") {
        paste("Dataset Name:", datasetName, shiny::a("Read Paper", href = "https://pubmed.ncbi.nlm.nih.gov/35675435/"))
      }
    })
    
    
    if (option == "Loomba et al.") {
      checkboxGroupInput("datasetCheckboxes_EV", "Select Dataset",
                         choices = c("Stool Species_External", "Stool Genus_External")
      )
    } else if (option == "Leung et al.") {
      checkboxGroupInput("datasetCheckboxes_EV", "Select Dataset",
                         choices = c("Stool Species-External", "Stool Genus-External")
      )
    }
  })
  
  
  
  # Event handler for selecting datasets
  observeEvent(input$datasetCheckboxes, {
    selectedDatasetIndices <- as.numeric(gsub("Dataset", "", input$datasetCheckboxes))
    selectedDatasets(datasetList())
    if (!is.null(selectedDatasets) && length(selectedDatasets) > 0) {
      selectedDataset(selectedDatasets)
    } else {
      selectedDataset(NULL)
    }
  })
  
  # Event handler for selecting external validation datasets
  observeEvent(input$datasetCheckboxes_EV, {
    selectedDatasetIndices <- as.numeric(gsub("Dataset", "", input$datasetCheckboxes_EV))
    selectedDatasets(datasetList())
    if (!is.null(selectedDatasets) && length(selectedDatasets) > 0) {
      selectedDataset(selectedDatasets)
    } else {
      selectedDataset(NULL)
    }
  })
  
  # Event handler for removing zero values
  observeEvent(input$removeZeroButton, {
    req(selectedDatasets())
    dataset <- datasets()
    
    
    cleanedDatasets <- lapply(selectedDatasets(), function(dataset) {
      zeroCols <- apply(dataset[, -1], 2, function(col) sum(col == 0, na.rm = TRUE))
      zeroThreshold <- input$removeZeroSlider/100 * nrow(dataset)
      dataset[, c(TRUE, zeroCols < zeroThreshold), drop = FALSE]
    })
    
    cleanedDatasetList(cleanedDatasets)
    selectedDatasets(cleanedDatasetList())
    processText("Zero Values Removed:")
  })
  
  # Event handler for removing missing values
  observeEvent(input$removeMissingButton, {
    req(selectedDatasets())
    dataset <- datasets()
    
    cleanedDatasets <- lapply(selectedDatasets(), function(dataset) {
      missingCols <- colSums(is.na(dataset[, -1]))
      missingThreshold <- input$removeMissingSlider/100 * nrow(dataset)
      dataset[, c(TRUE, missingCols <= missingThreshold), drop = FALSE]
    })
    cleanedDatasetList(cleanedDatasets)
    selectedDatasets(cleanedDatasetList())
    processText("Missing Values Removed:")
  })
  
  
  
  
  observeEvent(input$resetBtn, {
    session$reload()  # Restart the app
    
  })
  
  
  # Event handler for normalization
  observeEvent(input$normalizeButton, {
    req(cleanedDatasetList())
    method <- input$normalizationMethod
    if (method == "CPM+Log") {
      # Create a function to log-transform a matrix, handling missing values
      log_transform_with_missing <- function(X) {
        if (any(is.na(X))) {
          non_missing_values <- X[!is.na(X)]
          X[!is.na(X)] <- log((non_missing_values / sum(non_missing_values)) * 1e6 + 0.01, 10)
        } else {
          X <- log((X / sum(X)) * 1e6 + 0.01, 10)
        }
        return(X)
      }
      
      # Assuming cleanedDatasetList() returns a list of matrices or data frames
      
      
      # Apply the log transformation function to each dataset in the list
      normalizedDatasets <- lapply(cleanedDatasetList(), function(dataset) {
        dataset[, -1] <- sapply(dataset[, -1], log_transform_with_missing)
        dataset
      })
      
      
      
      cleanedDatasetList(normalizedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("CPM+Log Transformed Data")
    } else if (method == "Min-Max") {
      #dataset[, -1] <- scale(dataset[, -1], center = FALSE, scale = apply(dataset[, -1], 2, max) - apply(dataset[, -1], 2, min))
      # Create a function to normalize a vector, handling missing values
      normalize_with_missing <- function(x) {
        if (any(is.na(x))) {
          non_missing_values <- x[!is.na(x)]
          x[!is.na(x)] <- (non_missing_values - min(non_missing_values)) / (max(non_missing_values) - min(non_missing_values))
        } else {
          x <- (x - min(x)) / (max(x) - min(x))
        }
        return(x)
      }
      
      # Apply the normalization function to each dataset in the list
      normalizedDatasets <- lapply(cleanedDatasetList(), function(dataset) {
        dataset[, -1] <- sapply(dataset[, -1], normalize_with_missing)
        dataset
      })
      cleanedDatasetList(normalizedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("Min-Max Normalized Data")
      
    } else if (method == "Scale") {
      
      scale_normalize_with_missing <- function(x) {
        if (any(is.na(x))) {
          non_missing_values <- x[!is.na(x)]
          x[!is.na(x)] <- (non_missing_values - mean(non_missing_values, na.rm = TRUE)) / sd(non_missing_values, na.rm = TRUE)
        } else {
          x <- (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
        }
        return(x)
      }
      
      
      # Apply the scale normalization function to each dataset in the list
      normalizedDatasets <- lapply(cleanedDatasetList(), function(dataset) {
        dataset[, -1] <- sapply(dataset[, -1], scale_normalize_with_missing)
        dataset
      })
      
      cleanedDatasetList(normalizedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("Min-Max Normalized Data")
      
    } else if (method == "CPM") {
      # Create a function to perform CPM normalization on a vector, handling missing values
      cpm_normalize_with_missing <- function(x) {
        non_missing_sum <- sum(x, na.rm = TRUE)  # Calculate sum of non-missing values
        x <- ifelse(is.na(x), NA, x / non_missing_sum * 1000000)
        return(x)
      }
      
      # Apply the CPM normalization function to each dataset in the list
      normalizedDatasets <- lapply(cleanedDatasetList(), function(dataset) {
        dataset[, -1] <- apply(dataset[, -1], 2, cpm_normalize_with_missing)
        dataset
      })
      
      
      cleanedDatasetList(normalizedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("CPM Normalized Data")
      
    } else {
      selectedDatasets(cleanedDatasetList())
      processText("Data Not Normalized")
    }
  })
  
  # Event handler for imputation
  observeEvent(input$imputeButton, {
    req(cleanedDatasetList())
    method <- input$imputationMethod
    
    if (method == "kNN") {
      # k-Nearest Neighbors imputation
      library(class)
      k <- input$kValue
      imputedDatasets <- lapply(cleanedDatasetList(), function(dataset) {
        # Apply kNN imputation to each dataset
        imputed <- knnImputation(dataset[, -1], k = k)
        n = ncol(imputed) / 2
        dataset[, -1] <- imputed[, 1:n]
        dataset
      })
      cleanedDatasetList(imputedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("kNN Imputed Data")
      
    } else if (method == "Mean") {
      # Mean imputation and handling missing values
      
      A <- cleanedDatasetList()
      
      imputedDatasets <- lapply(A, function(dataset) {
        missing_columns <- colSums(is.na(dataset[, -1])) > 0
        
        if (any(missing_columns)) {
          dataset[, missing_columns] <- lapply(dataset[, missing_columns], function(column) {
            if (is.numeric(column)) {
              # Check for missing values in numeric columns
              if (any(is.na(column))) {
                # Replace missing values with the mean of the column
                column[is.na(column)] <- mean(column, na.rm = TRUE)
              }
            } else {
              # Check for missing values in non-numeric columns
              if (any(is.na(column))) {
                # Replace missing values with the most frequent value
                column[is.na(column)] <- names(sort(table(column), decreasing = TRUE))[1]
              }
            }
            column
          })
        }
        
        dataset
      })
      
      
      
      # Update cleanedDatasetList with imputed datasets
      cleanedDatasetList(imputedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("Mean Imputed Data")
      
      
    } else if (method == "MICE") {
      # Multiple Imputation by Chained Equations (MICE)
      imputedDatasets <- lapply(cleanedDatasetList(), function(dataset) {
        
        
        # imputed <- mice(dataset[, -1])
        # dataset[, -1] <- complete(imputed)
        # dataset
        
        
        # Set the number of imputations
        num_imputations <- 5  # Adjust as needed
        
        # Set the memory limit for each imputation
        memory_limit <- 10000  # Adjust based on available memory in MB
        
        # Identify columns with missing values
        cols_with_missing <- colnames(dataset[, -1])[colSums(is.na(dataset[, -1])) > 0]
        k=5
        chunk_cols=list()
        withProgress(message = 'Performing imputation...', value = 0, {
          # Loop through chunks of columns with missing values
          for (i in seq(1, length(cols_with_missing)-k, by = k)) {
            chunk_cols[[i]] <- cols_with_missing[i:(i+k)]  # Choose 5 columns at a time
            setProgress(message = paste('Performing imputation:', sprintf("%.2f%%", (i/length(cols_with_missing))*100)))
            
            if (length(cols_with_missing)-i==k+1)
            {
              chunk_cols[[i]] <- cols_with_missing[i:(i+k+1)]  # Choose 5 columns at a time
            }
            
            # Subset the dataframe to include only columns with missing values
            subset_df <- dataset[, -1][, chunk_cols[[i]]]
            subset_df=data.frame(subset_df)
            # Perform imputation using mice on the selected columns with memory limit
            imputed_data <- suppressMessages({mice(subset_df, m = num_imputations, mem.max = memory_limit)})
            
            # Update imputed values back into the main dataframe
            dataset[, -1][, chunk_cols[[i]]] <- complete(imputed_data)
            
            if (length(cols_with_missing)-(i+k)<k && length(cols_with_missing)-(i+k)>=2)
            {
              w=length(cols_with_missing)-(i+k)
              # Assuming cols_with_missing is your character vector with column names
              chunk_cols[[i]] <- cols_with_missing[(length(cols_with_missing) - w+1):length(cols_with_missing)]
              subset_df <- dataset[, -1][, chunk_cols[[i]]]
              imputed_data <- suppressMessages({mice(subset_df, m = num_imputations, mem.max = memory_limit)})
              dataset[, -1][, chunk_cols[[i]]] <- complete(imputed_data)
            }
            setProgress(value = i / length(cols_with_missing))
            
          }
          
          imputedDatasets <- knnImputation(dataset[, -1], k = 5)
          dataset[, -1]  <- imputedDatasets
          dataset
        })
      })
      
      cleanedDatasetList(imputedDatasets)
      selectedDatasets(cleanedDatasetList())
      processText("MICE Imputed Data")
      
    } else {
      # No imputation method selected
      selectedDatasets(cleanedDatasetList())
      processText("Data Not Imputed")
    }
  })
  
  
  processData <- function() {
    req(cleanedDatasetList())
    
    # Perform feature selection based on the selected method
    leftLabels <- input$leftLabelSelection
    rightLabels <- input$rightLabelSelection
    
    # Combine the selected labels
    selectedLabels <- c(leftLabels, rightLabels)
    
    # Check if "HCC=[LN+LX]" is in selectedLabels
    if ("HCC=[LN+LX]" %in% selectedLabels) {
      # Replace "HCC=[LN+LX]" with "HCC" in selectedLabels
      selectedLabels <- ifelse(selectedLabels == "HCC=[LN+LX]", "HCC", selectedLabels)
      
      # Replace "LN" and "LX" with "HCC" in the Label column (assuming it's the first column)
      P <- lapply(selectedDatasets(), function(sublist) {
        if ("Label" %in% names(sublist)) {
          sublist$Label[sublist$Label %in% c("LN", "LX")] <- "HCC"
        } else {
          sublist[, 1][sublist[, 1] %in% c("LN", "LX")] <- "HCC"
        }
        sublist
      })
    } else {
      # If "HCC=[LN+LX]" is not selected, proceed without modifications
      P <- selectedDatasets()
    }
    
    # Filter the dataset based on the selected labels
    filteredDataset <- lapply(P, function(sublist) {
      if ("Label" %in% names(sublist)) {
        subset <- sublist[sublist$Label %in% selectedLabels, ]
      } else {
        subset <- sublist[sublist[, 1] %in% selectedLabels, , drop = FALSE]
      }
      t(subset)
    })
    
    # Combine the filtered datasets using rbind
    combinedDataset <- do.call(rbind, filteredDataset)
    
    # Transpose the combined dataset
    combinedDataset <- t(combinedDataset)
    # Convert to data frame
    combinedDataset <- as.data.frame(combinedDataset)
    labelColumns <- which(colnames(combinedDataset) == "Label")
    if (length(labelColumns) > 1 && labelColumns[1] == 1) {
      labelColumns <- labelColumns[-1]
      combinedDataset <- combinedDataset[, -labelColumns]
    }
    
    datatable(combinedDataset)
    y=combinedDataset[,1]
    labelColumns <- which(colnames(combinedDataset) == "Label")
    combinedDataset <- combinedDataset[, -labelColumns]
    X <- combinedDataset
    
    List=list(X = X, y = y, leftLabels = leftLabels, rightLabels = rightLabels,combinedDataset=combinedDataset)
    return(List)
  }
  
  
  
  
  # Render the reactable of the selected dataset
  output$selectedDatasetTable <- renderReactable({
    
    req(selectedDatasets())
    dataset=datasets()
    reactable::reactable(as.data.frame(selectedDatasets()))
    
    if (!is.null(selectedDatasets())) {
      List=processData()
      y=List$y
      X=List$X
      L <- data.frame(Label = y)
      D=cbind(L,X)
      datatable(D)
      
      num_rows <- 10
      num_cols <- 50
      
      # Display the data table with desired rows and columns
      DT::datatable(D, options = list(pageLength = num_rows, scrollX = TRUE, scrollY = TRUE,
                                      autoWidth = TRUE, columnDefs = list(list(width = '50px',
                                                                               targets = "_all"))),
                    extensions = c("FixedColumns"), style = "bootstrap", class = "compact")
      
    }
  })
  
  # Render the text for the data processing information
  output$processText <- renderText({
    processText()
  })
  
  # Render the text for the dimensions of the selected dataset
  output$dimText <- renderText({
    req(selectedDatasets())
    req(datasets())
    dataset=datasets()
    
    if (!is.null(selectedDataset())) {
      List=processData()
      X=List$X
      dimensions <- dim(X)
      
      
      paste(dimensions[1], "Samples,", dimensions[2] , "Features")
    } else {
      ""
    }
  })
  
  # Render the text for the selected groups
  output$groupText <- renderText({
    if (!is.null(selectedDataset())) {
      # Get the selected labels from the checkbox inputs
      leftLabels <- input$leftLabelSelection
      rightLabels <- input$rightLabelSelection
      
      # Create separate strings for left and right labels
      leftLabelsText <- paste(leftLabels, collapse = " ")
      rightLabelsText <- paste(rightLabels, collapse = " ")
      
      # Combine the left and right labels with "vs"
      groupText <- paste("Groups:", leftLabelsText, "vs", rightLabelsText)
      
      groupText
    } else {
      ""
    }
  })
  
  # Reactive expression for dataset name
  datasetName <- reactive({
    req(selectedDataset())
    if (!is.null(selectedDataset())) {
      datasetNames <- sapply(seq_along(selectedDataset()), function(i) {
        combinedNames <- allCombinedNames()
        splitString <- unlist(strsplit(combinedNames[i], " "))
        paste("Dataset", i, ":", splitString[2])
      })
      paste(datasetNames, collapse = " ")
    } else {
      ""
    }
  })
  
  
  
  # Render the summary output
  output$summary_output <- renderPrint({
    req(selectedDatasets())
    if (!is.null(selectedDatasets()) && !is.null(input$datasetCheckboxes)) {
      List=processData()
      X=List$X
      dimensions <- dim(X)
      max_obs <- dimensions[1]
      updateSliderInput(session, "obs_slider", max = max_obs)
    } else {
      ""
    }
    selected_obs <- input$obs_slider
  })
  
  # Observe event for obs_slider
  observeEvent(input$obs_slider, {
    if (!is.null(selectedDataset())) {
      # Get the current observation index
      current_obs <- input$obs_slider
      List=processData()
      X=List$X      
      Xq <- sapply(X, as.numeric)
      current_data <- Xq[current_obs, ]
      
      # Compute the summary of the current data
      summary_text <- paste("Summary of Observation", current_obs, ":\n")
      summary_text <- paste(summary_text, "Mean:", round(mean(current_data, na.rm = TRUE), 2), "\n")
      summary_text <- paste(summary_text, "Median:", round(median(current_data, na.rm = TRUE), 2), "\n")
      summary_text <- paste(summary_text, "Min:", round(min(current_data, na.rm = TRUE), 2), "\n")
      summary_text <- paste(summary_text, "Max:", round(max(current_data, na.rm = TRUE), 2), "\n")
      summary_text <- paste(summary_text, "Standard Deviation:", round(sd(current_data, na.rm = TRUE), 2), "\n")
      summary_text <- paste(summary_text, "75th Percentile:", round(quantile(current_data, 0.75, na.rm = TRUE), 2), "\n")
      summary_text <- paste(summary_text, "Number of Zero Values:", sum(current_data == 0), "\n")
      summary_text <- paste(summary_text, "Number of Missing Values:", sum(is.na(current_data)), "\n")
      
      # Update the summary output
      output$summary_output <- renderPrint({
        cat(summary_text)
      })
      
      # Render the stat plot
      output$stat_plot <- renderPlot({
        par(mfrow = c(4, 2))  # Set subplot layout
        
        param_values1 <- round(sapply(1:input$obs_slider, function(obs) mean(Xq[obs, ], na.rm = TRUE)), 2)
        param_values2 <- round(sapply(1:input$obs_slider, function(obs) median(Xq[obs, ], na.rm = TRUE)), 2)
        param_values3 <- round(sapply(1:input$obs_slider, function(obs) min(Xq[obs, ], na.rm = TRUE)), 2)
        param_values4 <- round(sapply(1:input$obs_slider, function(obs) max(Xq[obs, ], na.rm = TRUE)), 2)
        param_values5 <- round(sapply(1:input$obs_slider, function(obs) sd(Xq[obs, ], na.rm = TRUE)), 2)
        param_values6 <- round(sapply(1:input$obs_slider, function(obs) quantile(Xq[obs, ],0.25, na.rm = TRUE)), 2)
        param_values7 <- round(sapply(1:input$obs_slider, function(obs) sum(Xq[obs, ] == 0, na.rm = TRUE)), 2)
        param_values8 <- round(sapply(1:input$obs_slider, function(obs) sum(is.na(Xq[obs, ]), na.rm = TRUE)), 2)
        
        plot(1:input$obs_slider, param_values1, type = "l",
             main = "Trend of mean", xlab = "Observation", ylab = "Mean Values")
        plot(1:input$obs_slider, param_values2, type = "l",
             main = "Trend of median", xlab = "Observation", ylab = "Median Values")
        plot(1:input$obs_slider, param_values3, type = "l",
             main = "Trend of min", xlab = "Observation", ylab = "min Values")
        plot(1:input$obs_slider, param_values4, type = "l",
             main = "Trend of max", xlab = "Observation", ylab = "max Values")
        plot(1:input$obs_slider, param_values5, type = "l",
             main = "Trend of Standard Deviation", xlab = "Observation", ylab = "max Values")
        plot(1:input$obs_slider, param_values7, type = "l",
             main = "Number of Zero Values:", xlab = "Observation", ylab = "max Values")
        plot(1:input$obs_slider, param_values6, type = "l",
             main = "Trend of 75th Percentile", xlab = "Observation", ylab = "max Values")
        plot(1:input$obs_slider, param_values8, type = "l",
             main = "Trend of Missing Values", xlab = "Observation", ylab = "max Values")
        
        if (input$obs_slider > 1) {
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values1[input$obs_slider-1], param_values1[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values2[input$obs_slider-1], param_values2[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values3[input$obs_slider-1], param_values3[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values4[input$obs_slider-1], param_values4[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values5[input$obs_slider-1], param_values5[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values6[input$obs_slider-1], param_values6[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values7[input$obs_slider-1], param_values7[input$obs_slider]), col = "red")
          lines(c(input$obs_slider-1, input$obs_slider), c(param_values8[input$obs_slider-1], param_values8[input$obs_slider]), col = "red")
        }
      })
    }
  })
  
  # Reactive function to generate the plot based on the selected method
  generatePlot <- reactive({
    method <- input$methodSelect
    List=processData()
    y=List$y
    X=List$X
    L <- data.frame(Label = y)
    leftLabels=List$leftLabels
    rightLabels=List$rightLabels
    selectedLabels=c(leftLabels, rightLabels)
    labelColumns <- which(colnames(X) == "Label")
    
    
    if (length(labelColumns) > 0) {
      X1 <- X[, -labelColumns, drop = FALSE]
    } else {
      X1 <- X
    }
    
    X1 <- sapply(X1, as.numeric)
    
    if (method == "t-SNE") {
      if (!is.null(selectedDatasets())) {
        
        colors <- rainbow(length(selectedLabels))
        names(colors) <- unique(selectedLabels)
        L <- length(selectedLabels)
        set.seed(123)
        tsne <- Rtsne(X1, dims = 2, perplexity = 15, check_duplicates = FALSE, verbose = TRUE, max_iter = 500)
        A <- plot(tsne$Y, col = "blue", bg = colors, pch = 21, cex = 1.5)
        title(paste("t-SNE:", allCombinedNames()))
        legend("topright", selectedLabels, col = 1:L, cex = 0.8, fill = 1:L)
        A + theme(plot.background = element_rect(fill = "white"))
        Plot(A)
      }  
      
    } else if (method == "UMAP") {
      if (!is.null(selectedDatasets())) {
        
        colors = rainbow(length(selectedLabels))
        names(colors) = unique(selectedLabels)
        L = length(selectedLabels)
        umap <- umap(X1,  n_neighbors= 15)
        plot=plot(umap$layout, col = "blue", bg = colors, pch = 21, cex = 1.5)
        title(paste("UMAP:",allCombinedNames()))
        legend("topright", selectedLabels, col = 1:L, cex = 0.8, fill = 1:L)
      }
      
    } else if (method == "PCA") {
      if (!is.null(selectedDatasets())) {
        
        # Perform PCA
        pca <- prcomp(X1, scale. = TRUE)
        # Extract the principal components
        pc <- pca$x
        # Create a data frame with PC1, PC2, and Label columns
        df <- data.frame(PC1 = pc[, 1], PC2 = pc[, 2], Label = L$Label)
        # Create a color palette for the labels
        colors <- rainbow(length(selectedLabels))
        names(colors) <- unique(selectedLabels)
        L <- length(selectedLabels)
        
        # Plot the principal components using ggplot2
        plot <- ggplot(df, aes(x = PC1, y = PC2, fill = Label)) +
          geom_point(color = "blue", shape = 21, size = 3.5) +
          scale_fill_manual(values = colors) +
          xlab("PC1") +
          ylab("PC2") +
          ggtitle(paste("PCA:", allCombinedNames())) +
          theme_bw() +
          theme(plot.background = element_rect(fill = "white"))
        
        # Add the legend
        plot <- plot + guides(fill = guide_legend(override.aes = list(color = 1:L, fill = 1:L),
                                                  title = "Label",
                                                  label.theme = element_text(size = 8),
                                                  keywidth = 0.8, keyheight = 0.8))
        # Return the plot
        return(plot)
      }
      
    } else if (method == "Diversity") {
      diversityType <- NULL
      
      if (input$simpsonCheckbox) {
        diversityType <- "simpson"
      } else if (input$shannonCheckbox) {
        diversityType <- "shannon"
      }
      
      if (!is.null(selectedDatasets()) && !is.null(diversityType)) {
        
        L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
        df <- data.frame(X1, L)
        feature_cols <- df[, 1:ncol(df) - 1] 
        diversity_values <- apply(feature_cols, 1, function(x) diversity(x, index = diversityType))
        df$Diversity <- diversity_values
        rightLabels <- paste(rightLabels, collapse = "_")
        df$Label <- factor(df$Label, levels = c(0, 1), labels = c(leftLabels, rightLabels))
        Group <- factor(df$Label)
        
        plot <- ggplot(df, aes(x = factor(df$Label), y = Diversity, fill = Group)) +
          geom_boxplot(outlier.shape = NA, notch = TRUE, width = 0.6) +
          scale_fill_manual(values = c("blue", "red")) +
          xlab("Group") +
          ylab(paste(diversityType, "diversity")) +
          ggtitle(allCombinedNames()) +
          theme_bw()
        plot + theme(plot.background = element_rect(fill = "white"))
      }
    }
  })
  
  
  # Render the plot output
  output$visualizationPlot <- renderPlot({
    generatePlot()
  })
  
  # Define the download handler for saving the plot
  output$DownloadPlot <- downloadHandler(
    filename = function() {
      paste(input$methodSelect, "plot", Sys.Date(), ".png", sep = "_")
    },
    content = function(file) {
      # Save the plot as JPG
      tmp_file <- tempfile(fileext = ".jpg")
      png(tmp_file, width = 800, height = 600, units = "px")
      print(generatePlot())
      dev.off()
      # Copy the temporary file as a JPG file
      file.copy(tmp_file, file)
    }
  )
  
  
  observeEvent(input$featureSelectionButton, {
    
    if (input$featureSelectionMethod == "Ranger") {
      message <- "Ranger Feature Selection Method"
      req(selectedDataset())
      
      if (input$impurity) {
        imp <- "impurity"
      }
      
      if (input$permutation) {
        imp <- "permutation"
      }
      
      if (!is.null(selectedDatasets())) {
        List=processData()
        y=List$y
        X=List$X
        
        leftLabels=List$leftLabels
        rightLabels=List$rightLabels
        L <- data.frame(Label = y)
        k <- input$kInput
        selectedLabels=c(leftLabels, rightLabels)
        if ("HCC=[LN+LX]" %in% selectedLabels) {
          # Replace "HCC=[LN+LX]" with "HCC" in selectedLabels
          selectedLabels <- ifelse(selectedLabels == "HCC=[LN+LX]", "HCC", selectedLabels)
        }
        all_labels <- unique(selectedLabels)
        class_mapping <- setNames(seq_along(all_labels) - 1, all_labels)
        L$Label <- as.integer(ifelse(L$Label %in% names(class_mapping), class_mapping[as.character(L$Label)], NA))
        classes <- unique(L$Label)
        
        data <- X
        Label <- L
        
        top_features <- list()
        feature_freq_df <- list()
        acc <- list()
        Metric <- list()
        Metric_valid <- list()
        M <- data.frame()
        PX <- data.frame()
        VP <- data.frame()
        NP <- data.frame()
        P_TOP <- data.frame()
        NP_TOP <- data.frame()
        VP_TOP <- data.frame()
        kk <- 0
        TOP <- input$numTOP
        iterations <- input$numIterations
        iterations=iterations/5
        
        withProgress(message = 'Performing iterations...', value = 0, {
          
          for (j in 1:iterations) {
            
            folds <- createFolds(Label$Label, k = 5, returnTrain = FALSE)
            
            for (i in 1:length(folds)) {
              kk <- kk + 1
              
              setProgress(message = paste('Iteration:', kk))
              
              # Split data into train and test sets
              train_data <- data[-folds[[i]], ]
              train_label <- Label$Label[-folds[[i]]]
              L <- length(train_data)
              
              
              #Apply feature selection using ranger
              model_ranger <- ranger::ranger(x = train_data, y = train_label, importance = imp)
              features <- which(model_ranger$variable.importance != 0)
              varImp <- data.frame(model_ranger[["variable.importance"]])
              VarImp <- data.frame(varImp[, 1])[, 1]
              P <- order(VarImp, decreasing = TRUE)
              PX[kk, 1:L[1]] <- t(P)
              NP[kk, 1:L[1]] <- t(rownames(data.frame(model_ranger$variable.importance))[P])
              VP[kk, 1:L[1]] <- t(data.frame(model_ranger$variable.importance[t(PX[i, 1:L[1]])]))
              P_TOP[kk, 1:TOP] <- PX[kk, 1:L[1]][1:TOP]
              NP_TOP[kk, 1:TOP] <- NP[kk, 1:L[1]][1:TOP]
              VP_TOP[kk, 1:TOP] <- VP[kk, 1:L[1]][1:TOP]
              
            }   
            
            setProgress(value = (kk / (iterations * k)))
            
          }
        })
        
        
        # Calculate frequency of each feature
        feature_freq <- table(unlist(NP_TOP)) 
        feature_freq_df <- data.frame(feature = names(feature_freq), frequency = feature_freq)
        # Filter features that appear in over 80% of iterations
        top_features <- feature_freq_df[feature_freq_df$frequency.Freq >= (input$percentageTOP/100)*input$numIterations, ]
        selected_features <- top_features$feature
        selected_features_freq <- top_features$frequency.Freq
        best_selected_features <- data.frame(
          feature = selected_features,
          frequency = as.numeric(selected_features_freq)
        )
        best_selected_features <- best_selected_features %>% arrange(frequency)
        best_selected_features_mean_sd <- best_selected_features %>%
          group_by(feature) %>%
          summarise(mean_frequency = mean(frequency), sd_frequency = sd(frequency)) %>% 
          arrange(mean_frequency)
        data_sel <- data[, best_selected_features_mean_sd$feature]
        BFS <- data.frame(features = best_selected_features_mean_sd$feature, mean_freq = best_selected_features_mean_sd$mean_frequency/input$numIterations)
        selectedFeaturesOutput=BFS$features
        selectedFeaturesOutput(selectedFeaturesOutput)
        output$selectedFeaturesOutput <- renderText({
          paste(selectedFeaturesOutput, collapse = ", ")
          
        })
        
        # 
        # feature_names2 <- sub("^(\\w+)_.*", "\\1", unique(BFS$features))
        # unique_prefixes <- unique(sapply(strsplit(feature_names2, "_"), `[`, 1))
        # feature_names2=unique_prefixes
        # # Get unique feature groups
        # feature_groups <- unique(feature_names2)
        # # Assign colors to feature groups
        # num_groups <- length(feature_groups)
        # color_palette <- c("steelblue", "darkorange", "forestgreen", "purple", "red", "yellow","green","gray", "cyan", "magenta")[1:num_groups]
        # # Match each feature group to its corresponding color
        # feature_colors <- color_palette[match(feature_names2, feature_groups)]
        # # Define the legend labels
        # legend_labels <- ifelse(feature_groups == "SS", "Stool Species",
        #                         ifelse(feature_groups == "SG", "Stool Genus",
        #                                ifelse(feature_groups == "OS", "Oral Species",
        #                                       ifelse(feature_groups == "OG", "Oral Genus",
        #                                              ifelse(feature_groups == "Path", "Pathology",
        #                                                     ifelse(feature_groups == "Clin", "Clinical",
        #                                                            ifelse(feature_groups == "MET", "Metabolomic",
        #                                                                   ifelse(feature_groups == "Lip", "Lipoprotein",
        #                                                                          ifelse(feature_groups == "Mol", "Molecules",
        #                                                                   ifelse(feature_groups == "Cyt", "Cytokine", ""))))))))))
        # print(BFS)
        # # Features appearing in over k% of iterations
        # Title <- paste("Ranger Features with Over", input$percentageTOP, "% Iterative Appearance:", collapse = "-")
        # Subtitle <- allCombinedNames()
        # feature_names <- unique(BFS$features)  # Get unique feature names
        # ggplot_object <- ggplot(BFS, aes(x = mean_freq, y = reorder(features, mean_freq))) +
        #   geom_bar(stat = "identity", fill = feature_colors) +
        #   ggtitle(Title) +
        #   labs(subtitle = Subtitle) +
        #   xlab("Mean Frequency") +
        #   ylab("Feature") +
        #   theme_classic() +
        #   theme(
        #     legend.position = "right",
        #     legend.title = element_text(size = 14),
        #     legend.text = element_text(size = 12),
        #     axis.title.x = element_text(size = 14),
        #     axis.title.y = element_text(size = 14),
        #     axis.text.y = element_text(size = 14)
        #   ) +
        #   scale_y_discrete(labels = feature_names) +
        #   guides(fill = guide_legend(title = "Legend", labels = legend_labels))
        # 
        # ggplot_object
        # 
        # 
        feature_names2 <- sub("^(\\w+)_.*", "\\1", BFS$features)
        
        
        feature_names2 <- unique(sapply(strsplit(feature_names2, "_"), `[`, 1))
        # Define a mapping of prefixes to full names
        prefix_full_names <- c(
          "SS" = "Stool Species", "SG" = "Stool Genus", "OS" = "Oral Species", "OG" = "Oral Genus",
          "Path" = "Pathology", "Clin" = "Clinical", "MET" = "Metabolomic", "Lip" = "Lipoprotein",
          "Mol" = "Molecules", "Cyt" = "Cytokine", "SM"="Small Molecule"
        )
        
        # Map the full names to feature names
        feature_full_names <- prefix_full_names[unique(feature_names2)]
        unique_full_names <- full_names <- unlist(feature_full_names)
        
        # Create a color mapping based on prefixes
        # feature_colors <- c(
        #   "SS" = "red", "SG" = "blue", "OS" = "green", "OG" = "orange", "Path" = "yellow",
        #   "Clin" = "purple", "MET" = "pink", "Lip" = "gray", "Mol" = "cyan", "Cyt" = "magenta", "SM"="brown"
        # )
        feature_colors <- c(
          "SS" = "gray", "SG" = "gray", "OS" = "gray", "OG" = "gray", "Path" = "gray",
          "Clin" = "gray", "MET" = "gray", "Lip" = "gray", "Mol" = "gray", "Cyt" = "gray", "SM"="gray"
        )
        # Create a bar plot
        Title <- paste("Ranger Features with Over", input$percentageTOP, "% Iterative Appearance:", collapse = "-")
        feature_colores <- sub("^([^_]+).*", "\\1", BFS$features)
        
        ggplot_object <- ggplot(BFS, aes(x = mean_freq, y = reorder(features, mean_freq), fill = feature_colores)) +
          geom_bar(stat = "identity") +
          ggtitle(Title) +
          xlab("Mean Frequency") +
          ylab("Feature") +
          scale_fill_manual(values = feature_colors, name = "Module", labels = unique_full_names) +
          theme_minimal() +
          theme(
            legend.position = "right",
            legend.title = element_text(size = 14),
            legend.text = element_text(size = 12),
            axis.title.x = element_text(size = 14),
            axis.title.y = element_text(size = 14),
            axis.text.y = element_text(size = 12)
          )
        
        ggplot_object
        
        
        
        output$Featureplot <- renderPlot({
          print(ggplot_object)
        })
        rangerselectedfeatures(ggplot_object)
        
      }
      
      y=List$y
      X=List$X
      L <- data.frame(Label = y)
      selectedLabels=c(leftLabels, rightLabels)
      if ("HCC=[LN+LX]" %in% selectedLabels) {
        # Replace "HCC=[LN+LX]" with "HCC" in selectedLabels
        selectedLabels <- ifelse(selectedLabels == "HCC=[LN+LX]", "HCC", selectedLabels)
      }
      all_labels <- unique(selectedLabels)
      class_mapping <- setNames(seq_along(all_labels) - 1, all_labels)
      L$Label <- as.integer(ifelse(L$Label %in% names(class_mapping), class_mapping[as.character(L$Label)], NA))
      #selectedFeaturesOutput =c("MET_X5660", "MET_X502","MET_X497", "MET_X438","MET_X401",  "MET_X3969","MET_X3918","MET_X2632" ,"MET_X2591","MET_X2590" , "MET_X2584","MET_X2580" ,"MET_X2381",
      #"MET_X2144", "MET_X2093", "MET_X4216","MET_X374","MET_X2076","MET_X5682", "MET_X5652","MET_X2870","MET_X2216")
      X <- cbind(L, X[, selectedFeaturesOutput])
      X <- mutate_all(X, as.numeric)
      X$Label <- factor(X$Label)
      
      # Prepare the data for plotting
      plot_data <- reshape2::melt(X, id.vars = "Label")
      
      # Plot the figure using ggplot2
      plot <- ggplot(plot_data, aes(x = variable, y = pmax(0, value), fill = as.factor(Label), group = Label)) +
        geom_bar(stat = "identity", position = "dodge", width = 0.7) +  # Adjust position to "dodge"
        labs(x = "Features", y = "Abundance") +
        ggtitle(paste("Relative Abundances of Top Features:", allCombinedNames())) +
        scale_fill_manual(name = "Classes", values = c("0" = "yellow", "1" = "green", "2" = "red"), labels = names(class_mapping)) +
        scale_y_continuous(expand = c(0, 0)) +  # Ensure bars start from zero
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.title = element_text(size = 10),  # Adjust the font size of the legend title
          legend.key.size = unit(0.6, "cm"),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      output$Abundancesplot <- renderPlot({
        plot
      })
      
      
      
      
      
      
      # Define the download handler for saving the plot
      output$plotsave2 <- downloadHandler(
        filename = function() {
          paste(input$featureSelectionMethod, "Relative Abundances of Top Features", Sys.Date(), ".jpg", sep = "_")
        },
        content = function(file) {
          # Save the plot as a temporary file
          tmp_file <- tempfile(fileext = ".jpg")
          png(tmp_file, width = 800, height = 600, units = "px")
          plot(plot)
          dev.off()
          # Copy the temporary file as a JPG file
          file.copy(tmp_file, file)
        }
      )
      
      
    } else if (input$featureSelectionMethod == "Wilcoxon") {
      message <- "Wilcoxon Feature Selection Method"
      req(selectedDatasets())
      
      if (!is.null(selectedDatasets())) {
        
        List=processData()
        y=List$y
        X=List$X
        leftLabels=List$leftLabels
        rightLabels=List$rightLabels
        L <- data.frame(Label = y)
        L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
        L$Label <- factor(L$Label)
        data <- X
        Labell <- L
        # Add SampleID column as row names
        data$SampleID <- rownames(data)
        # Reshape data into long format
        data_long <- data %>% 
          pivot_longer(-SampleID, names_to = "Feature", values_to = "Abundance")
        # Apply log transformation to abundance
        data_long <- data_long %>% 
          mutate(log_abundance = log10(as.numeric(Abundance)+0.1))
        # Merge with label data
        Labell_df <- data.frame(SampleID = rownames(data), Labell)
        data_long <- left_join(data_long, Labell_df, by = "SampleID")
        # Wilcoxon rank sum test
        test_results <- data_long %>% 
          group_by(Feature) %>% 
          summarize(pval = wilcox.test(log_abundance ~ Label)$p.value) %>% 
          mutate(qval = p.adjust(pval, method = "fdr"))
        #Top differentially abundant features
        top_features <- test_results %>%
          filter(pval < 0.05) %>%
          top_n(input$numGroups, wt = -pval) %>%
          pull(Feature)
        # Top differentially abundant phyla
        top_phyla <- data_long %>% 
          mutate(Phylum = str_extract(Feature, "^[^_]++")) %>% 
          group_by(Phylum) %>% 
          summarize(mean_abundance = mean(log_abundance)) %>% 
          filter(Phylum %in% str_extract(top_features, "^[^_]++")) %>% 
          top_n(input$numGroups, wt = -mean_abundance) %>% 
          pull(Phylum)
        
        
        p <- NULL
        data_plot <- reactive({
          new_legend_labels <- c(leftLabels, rightLabels)
          df <- data.frame(Component = rightLabels, stringsAsFactors = FALSE)
          combined <- paste(df$Component, collapse = "_")
          data_long$Label <- ifelse(data_long$Label %in% 0, leftLabels, ifelse(data_long$Label %in% 1, combined, NA))
          p <- data_long %>% 
            mutate(Phylum = str_extract(Feature, "^[^_]++")) %>% 
            filter(Phylum %in% top_phyla) %>% 
            filter(Feature %in% top_features) %>% 
            ggboxplot(x = "Feature", y = "log_abundance", fill = "Label", 
                      color = "Label", palette = "jco", 
                      ggtheme = theme_pubr()) +
            stat_compare_means(method = "wilcox.test", label = "p.format", 
                               label.y = c(90, 10), label.x = c(1, 2), size = 6) +
            coord_cartesian(ylim = range(data_long$log_abundance)) +
            labs(x = paste("Top", input$numGroups, "Features"), y = "Abundance (log10)", fill = "Label") +
            theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
          p
          
        })
        
        selectedFeaturesOutput=top_features
        selectedFeaturesOutput(selectedFeaturesOutput)
        output$selectedFeaturesOutput <- renderText({
          paste(selectedFeaturesOutput, collapse = ", ")
        })
        
        output$Featureplot <- renderPlot({
          plot(data_plot())       
        })
      }
      wilcoxon_plot=data_plot()
      wilcoxonselectedfeatures(wilcoxon_plot)
      L <- data.frame(Label = y)
      L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
      X <- cbind(L, X[, selectedFeaturesOutput])
      X <- mutate_all(X, as.numeric)
      rightLabels <- paste(rightLabels, collapse = "_")
      X$Label <- ifelse(X$Label %in% 0,leftLabels, ifelse(X$Label %in% 1,rightLabels, NA))
      X$Label <- factor(X$Label)
      
      # Prepare the data for plotting
      plot_data <- reshape2::melt(X, id.vars = "Label")
      # Set the margin for the plot
      par(mar = c(5, 4, 4, 2) + 0.1)
      # Plot the figure
      plot <- ggplot(plot_data, aes(x = variable, y = value, fill = Label)) +
        geom_bar(stat = "identity", position = "dodge", width = 0.7) +
        labs(x = "Features", y = "Abundance") +
        ggtitle(paste("Relative Abundances of Top Features:", allCombinedNames())) +
        scale_fill_manual(values = c("red", "blue")) +
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.title = element_blank(),
          legend.key.size = unit(0.6, "cm"),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      output$Abundancesplot <- renderPlot({
        plot
      })
      
      
      # Define the download handler for saving the plot
      output$plotsave2 <- downloadHandler(
        filename = function() {
          paste(input$featureSelectionMethod, "Relative Abundances of Top Features", Sys.Date(), ".jpg", sep = "_")
        },
        content = function(file) {
          # Save the plot as a temporary file
          tmp_file <- tempfile(fileext = ".jpg")
          png(tmp_file, width = 800, height = 600, units = "px")
          plot(plot)
          dev.off()
          
          # Copy the temporary file as a JPG file
          file.copy(tmp_file, file)
        }
      )
      
      
    } else if (input$featureSelectionMethod == "Boruta") {
      # Code for Brouta feature selection method 
      message <- "Brouta Feature Selection Method"
      req(selectedDatasets())
      
      if (!is.null(selectedDatasets())) {
        
        List=processData()
        y=List$y
        X=List$X
        leftLabels=List$leftLabels
        rightLabels=List$rightLabels
        L <- data.frame(Label = y)
        k <- 5
        L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
        data <- X
        Label <- L
        NP_TOP <- list()
        kk=0
        iterations <- input$numIterationsB
        iterations=iterations/5
        withProgress(message = 'Performing iterations...', value = 0, {
          
          for (j in 1:iterations) {
            folds <- createFolds(Label$Label, k = 5, returnTrain = FALSE)
            for (i in 1:length(folds)) {
              kk <- kk + 1
              
              setProgress(message = paste('Iteration:', kk))
              
              # Split data into train and test sets
              train_data <- data[-folds[[i]], ]
              train_label <- Label$Label[-folds[[i]]]
              L <- length(train_data)
              classes <- c(0, 1)
              boruta_output <- suppressMessages(Boruta(x = train_data, y = factor(train_label),doTrace=2,maxRuns= 100, ntree=500,pValue=0.01))  
              NP=getSelectedAttributes(boruta_output,withTentative = F)
              NP_TOP[[kk]] <- NP
            }
            setProgress(value = (kk / (iterations * k)))
            
          }
        })
        # Calculate frequency of each feature across sublists
        feature_freq <- table(unlist(NP_TOP))
        # Filter features appearing in more than 70% of sublists
        selected_features <- names(feature_freq[feature_freq >= ((input$percentageTOPB)/100) * length(NP_TOP)])
        # Create data frame with selected features and frequencies
        selected_features_df <- data.frame(
          feature = selected_features,
          frequency = feature_freq[selected_features]
        )
        
        # Sort the data frame in descending order of frequencies
        selectedFeaturesOutput <- selected_features_df[order(selected_features_df$frequency.Freq, decreasing = TRUE), ]
        selectedFeaturesOutput=selectedFeaturesOutput$feature     
        # Create horizontal bar plot
        selectedFeaturesOutput(selectedFeaturesOutput)
        output$selectedFeaturesOutput <- renderText({
          paste(selectedFeaturesOutput, collapse = ", ")
        })
        
        # Plot feature selection results
        Title <- paste("Boruta Features with Over", input$percentageTOPB, "% Iterative Appearance:", collapse = "-")
        Subtitle <- allCombinedNames()
        
        ggplot_object=ggplot(selected_features_df, aes(x = frequency.Freq/input$numIterationsB, y = reorder(feature, frequency.Freq))) +
          geom_bar(stat = "identity", fill = "red") +
          ggtitle(Title) +
          labs(subtitle = Subtitle) +
          xlab("Mean Frequency") +
          ylab("Feature") +
          theme_classic() +
          theme(legend.position = "none",
                axis.title.x = element_text(size = 14),
                axis.title.y = element_text(size = 14),
                axis.text.y = element_text(size = 14)) +
          scale_y_discrete(labels = selectedFeaturesOutput)  # Use custom feature names on y-axis
        
        
        output$Featureplot <- renderPlot({
          print(ggplot_object)
        })
        
        broutaselectedfeatures(ggplot_object)
      }
      L <- data.frame(Label = y)
      L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
      X <- cbind(L, X[, selectedFeaturesOutput])
      X <- mutate_all(X, as.numeric)
      rightLabels <- paste(rightLabels, collapse = "_")
      X$Label <- ifelse(X$Label %in% 0,leftLabels, ifelse(X$Label %in% 1,rightLabels, NA))
      X$Label <- factor(X$Label)
      # Prepare the data for plotting
      plot_data <- reshape2::melt(X, id.vars = "Label")
      # Set the margin for the plot
      par(mar = c(5, 4, 4, 2) + 0.1)
      
      # Plot the figure
      plot <- ggplot(plot_data, aes(x = variable, y = value, fill = Label)) +
        geom_bar(stat = "identity", position = "dodge", width = 0.7) +
        labs(x = "Features", y = "Abundance") +
        ggtitle(paste("Relative Abundances of Top Features:", allCombinedNames())) +
        scale_fill_manual(values = c("red", "blue")) +
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.title = element_blank(),
          legend.key.size = unit(0.6, "cm"),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      output$Abundancesplot <- renderPlot({
        plot
      })
      
      
      # Define the download handler for saving the plot
      output$plotsave2 <- downloadHandler(
        filename = function() {
          paste(input$featureSelectionMethod, "Relative Abundances of Top Features", Sys.Date(), ".jpg", sep = "_")
        },
        content = function(file) {
          # Save the plot as a temporary file
          tmp_file <- tempfile(fileext = ".jpg")
          png(tmp_file, width = 800, height = 600, units = "px")
          plot(plot)
          dev.off()
          # Copy the temporary file as a JPG file
          file.copy(tmp_file, file)
        }
      )
      
    } else if (input$featureSelectionMethod == "Lefse") {
      # Code for Lefse feature selection method
      message <- "Lefse Feature Selection Method"
      req(selectedDatasets())
      # 
      if (!is.null(selectedDatasets())) {
        
        List=processData()
        y=List$y
        X=List$X
        leftLabels=List$leftLabels
        rightLabels=List$rightLabels
        L <- data.frame(Label = y)
        A<- mutate_all(X, as.numeric)
        columns_with_na <- colSums(is.na(A)) > 0
        A <- A[, !columns_with_na]
        df_transposed <- t(A)
        assay_matrix <- as.matrix(df_transposed)
        # Change labels to 0 and 1
        L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
        L$Label <- factor(L$Label, levels = c("1", "0"))
        # Create the SummarizedExperiment object
        col_data <- data.frame(
          label = L$Label,
          row.names = colnames(assay_matrix)
        )
        se <- SummarizedExperiment(assays = list(counts = assay_matrix), colData = col_data)
        # Perform Lefse analysis
        res_block <- lefser(se,input$kruskalthreshold,input$wilcox.threshold, groupCol = "label")
        selectedFeaturesOutput=res_block$Names
        selectedFeaturesOutput(selectedFeaturesOutput)
        output$selectedFeaturesOutput <- renderText({
          paste(selectedFeaturesOutput, collapse = ", ")
        })
        
        # Generate the Lefse plot
        lefsePlot <- lefserPlot(res_block, colors = c("red", "forestgreen"), trim.names = FALSE)
        legend_labels <- levels(lefsePlot$data$group)
        new_legend_labels <- c(leftLabels, rightLabels)
        df <- data.frame(Component = rightLabels, stringsAsFactors = FALSE)
        combined <- paste(df$Component, collapse = "_")
        legend_labels <- new_legend_labels[match(legend_labels, c("0", "1"))]
        lefsePlot$data$group <- factor(lefsePlot$data$group, levels = c("0", "1"), labels = c(leftLabels, combined))
        # Display the Lefse plot in the GUI
        output$Featureplot <- renderPlot({
          plot(lefsePlot)
        })
        lefseselectedfeatures(lefsePlot)
      }  
      L <- data.frame(Label = y)
      L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
      X <- cbind(L, X[, selectedFeaturesOutput])
      X <- mutate_all(X, as.numeric)
      rightLabels <- paste(rightLabels, collapse = "_")
      X$Label <- ifelse(X$Label %in% 0,leftLabels, ifelse(X$Label %in% 1,rightLabels, NA))
      X$Label <- factor(X$Label)
      # Prepare the data for plotting
      plot_data <- reshape2::melt(X, id.vars = "Label")
      # Set the margin for the plot
      par(mar = c(5, 4, 4, 2) + 0.1)
      # Plot the figure
      plot <- ggplot(plot_data, aes(x = variable, y = value, fill = Label)) +
        geom_bar(stat = "identity", position = "dodge", width = 0.7) +
        labs(x = "Features", y = "Abundance") +
        ggtitle(paste("Relative Abundances of Top Features:", allCombinedNames())) +
        scale_fill_manual(values = c("red", "blue")) +
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.title = element_blank(),
          legend.key.size = unit(0.6, "cm"),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      output$Abundancesplot <- renderPlot({
        plot
      })
      
      # Define the download handler for saving the plot
      output$plotsave2 <- downloadHandler(
        filename = function() {
          paste(input$featureSelectionMethod, "Relative Abundances of Top Features", Sys.Date(), ".jpg", sep = "_")
        },
        content = function(file) {
          # Save the plot as a temporary file
          tmp_file <- tempfile(fileext = ".jpg")
          png(tmp_file, width = 800, height = 600, units = "px")
          plot(plot)
          dev.off()
          # Copy the temporary file as a JPG file
          file.copy(tmp_file, file)
        }
      )
      
    } else if (input$featureSelectionMethod == "Random") {
      message <- "Random Feature Selection Method"
      req(selectedDatasets())
      
      if (!is.null(selectedDatasets())) {
        X=combinedDataset
        target <- data.frame(Label = y)
        
        # Select a subset of features randomly
        selected_features <- sample(colnames(combinedDataset), size = input$numfeatures)
        selectedFeaturesOutput=selected_features
        selectedFeaturesOutput(selectedFeaturesOutput)
        
        output$selectedFeaturesOutput <- renderText({
          paste(selectedFeaturesOutput, collapse = ", ")
        })
        
      }  
      L <- data.frame(Label = y)
      L$Label <- ifelse(L$Label %in% leftLabels, 0, ifelse(L$Label %in% rightLabels, 1, NA))
      X <- cbind(L, X[, selectedFeaturesOutput])
      X <- mutate_all(X, as.numeric)
      rightLabels <- paste(rightLabels, collapse = "_")
      X$Label <- ifelse(X$Label %in% 0,leftLabels, ifelse(X$Label %in% 1,rightLabels, NA))
      X$Label <- factor(X$Label)
      # Prepare the data for plotting
      plot_data <- reshape2::melt(X, id.vars = "Label")
      # Set the margin for the plot
      par(mar = c(5, 4, 4, 2) + 0.1)
      
      # Plot the figure
      plot <- ggplot(plot_data, aes(x = variable, y = value, fill = Label)) +
        geom_bar(stat = "identity", position = "dodge", width = 0.7) +
        labs(x = "Features", y = "Abundance") +
        ggtitle(paste("Relative Abundances of Top Features:", allCombinedNames())) +
        scale_fill_manual(values = c("red", "blue")) +
        theme_minimal() +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(color = "black"),
          legend.position = "top",
          legend.title = element_blank(),
          legend.key.size = unit(0.6, "cm"),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      
      output$Abundancesplot <- renderPlot({
        plot
      })
      
      
      # Define the download handler for saving the plot
      output$plotsave2 <- downloadHandler(
        filename = function() {
          paste(input$featureSelectionMethod, "Relative Abundances of Top Features", Sys.Date(), ".jpg", sep = "_")
        },
        content = function(file) {
          # Save the plot as a temporary file
          tmp_file <- tempfile(fileext = ".jpg")
          png(tmp_file, width = 800, height = 600, units = "px")
          plot(plot)
          dev.off()
          
          # Copy the temporary file as a JPG file
          file.copy(tmp_file, file)
        }
      )
      
      
    } else {
      # Handle other feature selection methods
      message <- "Unknown Feature Selection Method"
    }
    
    processText(message)
  })
  
  
  
  # Create a reactive expression to store the plot object
  selectedPlot <- reactive({
    
    switch(input$featureSelectionMethod,
           "Wilcoxon" = {
             wilcoxon_plot <- wilcoxonselectedfeatures()
             wilcoxon_plot
             
           },
           "Ranger" = {
             ggplot_object= rangerselectedfeatures()
             ggplot_object
           },
           "Boruta" = {
             ggplot_object= broutaselectedfeatures()
             ggplot_object
           },
           "Lefse" = {
             lefsePlot=lefseselectedfeatures()
             
             lefsePlot
           }
    )
  })
  
  
  
  # Define the download handler for saving the plot
  output$plotsave <- downloadHandler(
    filename = function() {
      paste(input$featureSelectionMethod, "plot", Sys.Date(), ".jpg", sep = "_")
    },
    content = function(file) {
      # Save the plot as a temporary file
      tmp_file <- tempfile(fileext = ".jpg")
      png(tmp_file, width = 800, height = 600, units = "px")
      plot(selectedPlot())
      dev.off()
      
      # Copy the temporary file as a JPG file
      file.copy(tmp_file, file)
    }
  )
  
  
  output$selectedFeaturesOutput <- renderText({
    method <- input$featureSelectionMethod
    
    selectedFeaturesOutput <- switch(
      method,
      "Wilcoxon" = top_features,  # Wilcoxon selected features
      "Ranger" = BFS$features,    # Ranger selected features
      "Boruta" = boruta_features, # Boruta selected features
      "Lefse" = res_block$Names   # Lefse selected features
    )
    
    paste(selectedFeaturesOutput, collapse = ", ")
  })
  
  
  
  
  observeEvent(input$classificationButton, {
    req(selectedDatasets(), selectedFeaturesOutput())
    List=processData()
    y=List$y
    X=List$X
    
    leftLabels=List$leftLabels
    rightLabels=List$rightLabels
    k <- input$kInput
    L <- data.frame(Label = y)
    selectedLabels=c(leftLabels, rightLabels)
    if ("HCC=[LN+LX]" %in% selectedLabels) {
      # Replace "HCC=[LN+LX]" with "HCC" in selectedLabels
      selectedLabels <- ifelse(selectedLabels == "HCC=[LN+LX]", "HCC", selectedLabels)
    }
    all_labels <- unique(selectedLabels)
    class_mapping <- setNames(seq_along(all_labels) - 1, all_labels)
    L$Label <- as.integer(ifelse(L$Label %in% names(class_mapping), class_mapping[as.character(L$Label)], NA))
    NG=length(all_labels)
    L$Label=factor(L$Label)
    folds <- createFolds(L$Label, k = k, returnTrain = FALSE)
    X=cbind(Label=L$Label,X)
    view(X)
    
    # Initialize vectors to store evaluation metrics
    sensitivity <- numeric(k)
    accuracy <- numeric(k)
    specificity <- numeric(k)
    f1score <- numeric(k)
    m=0
    cm=list()
    model=list()
    Outcome=data.frame()
    Sen=data.frame()
    Spe=data.frame()
    Acc=data.frame()
    F1=data.frame()
    
    # Perform k-fold cross-validation
    iterations <- input$numIterationsC
    iterations=iterations/5
    # Perform classification based on the selected method
    if (input$classificationMethod == "Random Forest") {
      message <- "Random Forest Classification Method"
      
      if (!is.null(selectedDatasets())) {
        
        withProgress(message = 'Performing iterations...', value = 0, {
          for (j in 1:iterations) {
            folds <- createFolds(L$Label, k = k, returnTrain = FALSE)
            
            for (i in 1:k) {
              m=m+1
              setProgress(message = paste('Iteration:', m))
              
              # Split the data into training and testing sets
              train_data <-X[-folds[[i]], ]
              test_data <- X[folds[[i]], ]
              
              # Prepare the input and label data for XGBoost
              x_train <- train_data[, -1][,selectedFeaturesOutput()]
              
              
              y_train <- train_data[, 1]
              x_test <- test_data[, -1][,selectedFeaturesOutput()]
              y_test <- test_data[, 1]
              y_train <- unlist(y_train)
              y_test <- unlist(y_test)
              y_train <- as.integer(y_train) - 1
              y_test <- as.integer(y_test) - 1
              x_train <- sapply(x_train, as.numeric)
              x_test <- sapply(x_test, as.numeric)
              
              # Train the Random Forest model
              model[[m]] <- train(x = x_train, y = y_train, method = "rf")
              # Make predictions on the test set
              pred <- predict(model[[m]], x_test)
              pred <- ifelse(pred > 0.5, 1, 0)
              
              pred=data.frame(values =pred)
              pred <- ifelse(pred$values == 1, 1, 0)
              
              # Evaluate the predictions
              y_test <- factor(y_test, levels = c(0, 1))
              pred <- factor(pred, levels = c(0, 1))
              
              # Calculate the confusion matrix
              cm[[m]] <- confusionMatrix(y_test, pred)
              # Extract evaluation metrics
              Outcome[m, 1:4] <- data.frame(
                specificity = cm[[m]][["byClass"]][["Specificity"]],
                sensitivity = cm[[m]][["byClass"]][["Sensitivity"]],
                accuracy = cm[[m]][["overall"]][["Accuracy"]],
                f1score = cm[[m]][["byClass"]][["F1"]]
              )
              setProgress(value = (m / (iterations * k)))
              
            }
          }
          
        })
        
      }
      
      
    } else if (input$classificationMethod == "GBM") {
      message <- "Random Forest Classification Method"
      
      if (!is.null(selectedDatasets())) {
        
        withProgress(message = 'Performing iterations...', value = 0, {
          for (j in 1:iterations) {
            folds <- createFolds(L$Label, k = k, returnTrain = FALSE)
            
            for (i in 1:k) {
              m=m+1
              setProgress(message = paste('Iteration:', m))
              # Split the data into training and testing sets
              
              train_data <-X[-folds[[i]], ]
              test_data <- X[folds[[i]], ]
              # Prepare the input and label data for XGBoost
              x_train <- train_data[, -1][,selectedFeaturesOutput()]
              y_train <- train_data[, 1]
              x_test <- test_data[, -1][,selectedFeaturesOutput()]
              y_test <- test_data[, 1]
              y_train <- unlist(y_train)
              y_test <- unlist(y_test)
              y_train <- as.integer(y_train) - 1
              y_test <- as.integer(y_test) - 1
              x_train <- sapply(x_train, as.numeric)
              x_test <- sapply(x_test, as.numeric)
              x_train=data.frame(x_train)
              y_train=as.factor(y_train)
              train_data=cbind(Label=y_train, x_train)
              x_test=data.frame(x_test)
              x_test=cbind(Label=y_test,x_test)
              x_test=data.frame(x_test)
              fitControl = trainControl(method="cv", number=5, returnResamp = "all")
              if (length(class_mapping) > 2) {
                # Multiclass classification
                model[[m]] <- train(Label ~ ., 
                                    data = train_data, 
                                    method = "gbm",
                                    distribution = "multinomial",  
                                    trControl = fitControl, 
                                    verbose = FALSE, 
                                    tuneGrid = data.frame(.n.trees = 100, .shrinkage = 0.01, .interaction.depth = 1, .n.minobsinnode = 1)
                )
              } else {
                # Binary classification
                model[[m]] <- train(Label ~ ., 
                                    data = train_data, 
                                    method = "gbm",
                                    distribution = "bernoulli",  
                                    trControl = fitControl, 
                                    verbose = FALSE, 
                                    tuneGrid = data.frame(.n.trees = 100, .shrinkage = 0.01, .interaction.depth = 1, .n.minobsinnode = 1)
                )
              }
              
              # Assuming pred and y_test are defined as described
              pred <- predict(object = model[[m]],
                              newdata = x_test[, -1], na.action = na.pass, type = "prob")
              
              # Convert probabilities to predicted class labels
              predicted_labels <- colnames(pred)[apply(pred, 1, which.max)]
              predicted_labels <- factor(predicted_labels, levels = levels(train_data$Label))
              
              # Evaluate the predictions
              y_test <- factor(y_test, levels = levels(predicted_labels))
              predicted_labels <- factor(predicted_labels, levels = levels(predicted_labels))
              
              # Calculate the confusion matrix using confusionMatrix function
              cm[[m]] <- confusionMatrix(y_test, predicted_labels)
              b=as.matrix(cm[[m]],what="classes")
              
              # Extract evaluation metrics
              Spe[m, 1:NG] = b[2,]
              Sen[m, 1:NG] = b[1,]
              Acc[m, 1:NG]= b[11,]
              F1[m, 1:NG]  = b[7,]
              
              # Outcome[m, 1:4] <- data.frame(
              #  
              # )
              
              
              setProgress(value = (m / (iterations * k)))
              
            }
          }
          
        })
        
      }
      
      
    } else if (input$classificationMethod == "SVM") {
      
      message <- "SVM Classification Method"
      
      if (!is.null(selectedDatasets())) {
        
        withProgress(message = 'Performing iterations...', value = 0, {
          
          for (j in 1:iterations) {
            folds <- createFolds(L$Label, k = k, returnTrain = FALSE)
            for (i in 1:k) {
              m=m+1
              setProgress(message = paste('Iteration:', m))
              
              # Split the data into training and testing sets
              train_data <-X[-folds[[i]], ]
              test_data <- X[folds[[i]], ]
              # Prepare the input and label data for SVM
              x_train <- train_data[, -1][,selectedFeaturesOutput()]
              y_train <- train_data[, 1]
              x_test <- test_data[, -1][,selectedFeaturesOutput()]
              y_test <- test_data[, 1]
              y_train <- unlist(y_train) 
              y_test <- unlist(y_test)  
              y_train <- as.integer(y_train) - 1
              y_test <- as.integer(y_test) - 1
              
              
              # Train the SVM model
              model[[m]] <- svm(x_train, y_train, probability = TRUE)
              
              # Make predictions on the test set
              pred <- predict(model[[m]], x_test)
              pred <- ifelse(pred > 0.5, 1, 0)
              
              pred=data.frame(values =pred)
              pred <- ifelse(pred$values == 1, 1, 0)
              
              # Evaluate the predictions
              y_test <- factor(y_test, levels = c(0, 1))
              pred <- factor(pred, levels = c(0, 1))
              
              # Calculate the confusion matrix
              cm[[m]] <- confusionMatrix(y_test, pred)
              
              
              # Extract evaluation metrics
              Outcome[m, 1:4] <- data.frame(
                specificity = cm[[m]][["byClass"]][["Specificity"]],
                sensitivity = cm[[m]][["byClass"]][["Sensitivity"]],
                accuracy = cm[[m]][["overall"]][["Accuracy"]],
                f1score = cm[[m]][["byClass"]][["F1"]]
              )
              setProgress(value = (m / (iterations * k)))
              
            }
          }
        })
        
      }
      
      
    } else if (input$classificationMethod == "XGBoost") {
      message <- "XGBoost Classification Method"
      
      if (!is.null(selectedDatasets())) {
        
        withProgress(message = 'Performing iterations...', value = 0, {
          
          for (j in 1:iterations) {
            folds <- createFolds(L$Label, k = k, returnTrain = FALSE)
            
            for (i in 1:k) {
              m=m+1
              setProgress(message = paste('Iteration:', m))
              
              # Split the data into training and testing sets
              train_data <-X[-folds[[i]], ]
              test_data <- X[folds[[i]], ]
              # Prepare the input and label data for XGBoost
              x_train <- train_data[, -1][,selectedFeaturesOutput()]
              
              y_train <- train_data[, 1]
              x_test <- test_data[, -1][,selectedFeaturesOutput()]
              y_test <- test_data[, 1]
              y_train <- unlist(y_train) 
              y_test <- unlist(y_test)  
              y_train <- as.integer(y_train) - 1
              y_test <- as.integer(y_test) - 1
              
              
              
              x_train <- as.data.frame(x_train)              # Convert the list to a data frame
              x_train <- sapply(x_train, as.numeric)
              x_train <- as.matrix(x_train)
              x_test <- as.data.frame(x_test)
              x_test <- sapply(x_test, as.numeric)
              x_test <- as.matrix(x_test)
              
              # Convert the data frame to a matrix
              xgb.train <- xgb.DMatrix(data = x_train, label = y_train)
              xgb.test <- xgb.DMatrix(data = x_test, label =y_test)
              
              # Check the number of unique labels
              num_labels <- length(unique(all_labels))
              
              # Define default parameters
              default_params <- list(
                booster = "gbtree",
                eta = 0.001,
                max_depth = 5,
                gamma = 3,
                subsample = 0.75,
                colsample_bytree = 1
              )
              
              # Set up parameters based on the number of labels
              if (num_labels == 2) {
                # Binary classification parameters
                params <- c(
                  default_params,
                  objective = "binary:logistic",
                  eval_metric = c("auc", "error")
                )
              } else {
                # Multiclass classification parameters
                params <- c(
                  default_params,
                  objective = "multi:softmax",
                  eval_metric = "mlogloss",
                  num_class = num_labels
                )
              }
              
              model[[m]] <- xgb.train(
                params = params,
                data = xgb.train,
                nrounds = 100,
                #nthreads = 1,
                early_stopping_rounds = 10,
                watchlist = list(train = xgb.train, test = xgb.test),
                verbose = 0
              )
              
              pred <- predict(model[[m]], as.matrix(x_test))
              # Adjust prediction based on the number of labels
              if (num_labels == 2) {
                pred <- ifelse(pred > 0.5, 1, 0)
              }
              # Evaluate the predictions
              cm[[m]] <- confusionMatrix(
                factor(y_test, levels = unique(y_test)),
                factor(pred, levels = unique(y_test))
              )
              
              
              # Extract evaluation metrics
              b=as.matrix(cm[[m]],what="classes")
              # Extract evaluation metrics
              Outcome[m, 1:4] <- data.frame(
                specificity = b[2,],
                sensitivity = b[1,],
                accuracy = b[11,],
                f1score = b[7,]
              )
              
              setProgress(value = (m / (iterations * k)))
              
            }
          }
        })
        
      }
      
      
    } else {
      # Handle other classification methods
      message <- "Unknown Classification Method"
    }
    classificationModel(model)
    
    # 
    # Results <- data.frame(
    #   #Iteration = rep(1:5, each = 5),
    #   Class = rep(levels(X$Label), times = iterations * 5*NG),
    #   Sensitivity = c(Sen[, 1], Sen[, 2],Sen[, 3]),
    #   Specificity = c(Spe[, 1], Spe[, 2],Spe[, 3] ),
    #   Accuracy = c(Acc[, 1], Acc[, 2],Acc[, 3]),
    #   F1Score = c(F1[, 1], F1[, 2],F1[, 3])
    # )
    
    
    
    
    X$Label <- factor(X$Label, levels = c(all_labels))
    
    
    if (NG>2){
      # Melt the data frame to long format
      df_melted <- melt(Sen)
      df_new1 <- data.frame(Sensitivity = df_melted$value)
      df_melted <- melt(Spe)
      df_new2 <- data.frame(Specificity = df_melted$value)
      df_melted <- melt(Acc)
      df_new3 <- data.frame(Accuracy = df_melted$value)
      df_melted <- melt(F1)
      df_new4 <- data.frame(F1Score = df_melted$value)
      
      S <- data.frame(
        Class <- rep(levels(X$Label), each = iterations *k)
      )
      Results=cbind(S,df_new1,df_new2,df_new3,df_new4)
      
      
      
      
      # Rename the columns with a prefix
      col_names <- c("Class","Sensitivity", "Specificity", "Accuracy", "F1Score")
      colnames(Results)=col_names
      
      
    }else{
      A <- cbind(Sensitivity = Sen[, 1],
                 Specificity = Spe[, 1],
                 Accuracy = Acc[, 1],
                 F1Score = F1[, 1])
      
      S <- data.frame(
        Class = c("CON","CIR","CON","CIR","CON")
      )
      Results=cbind(S,A)
      #Results=A
    }
    
    
    # for (col in col_names) {
    #   col_indices <- grep(col, names(Results))
    #   for (i in 1:NG) {
    #     Results[, col_indices[i]] <- paste0(col, "_", i)
    #   }
    # }
    # 
    # # Remove unnecessary columns
    # Results <- Results[, !(names(Results) %in% col_names)]
    # 
    # # View the resulting data frame
    # 
    
    
    
    #Results <- Results[!is.na(Results$Class), ]
    Outcome=data.frame(Results)
    
    
    # Calculate the mean and standard deviation of evaluation metrics
    mean_sensitivity <- mean(Outcome$Sensitivity,na.rm = TRUE) * 100
    sd_sensitivity <- sd(Outcome$Sensitivity,na.rm = TRUE) * 100
    mean_accuracy <- mean(Outcome$Accuracy,na.rm = TRUE) * 100
    sd_accuracy <- sd(Outcome$Accuracy,na.rm = TRUE) * 100
    mean_specificity <- mean(Outcome$Specificity,na.rm = TRUE) * 100
    sd_specificity <- sd(Outcome$Specificity,na.rm = TRUE) * 100
    mean_f1score <- mean(Outcome$F1Score,na.rm = TRUE) * 100
    sd_f1score <- sd(Outcome$F1Score,na.rm = TRUE) * 100
    
    
    # Format the values to two decimal places
    mean_sensitivity <- round(mean_sensitivity, 2)
    sd_sensitivity <- round(sd_sensitivity, 2)
    mean_accuracy <- round(mean_accuracy, 2)
    sd_accuracy <- round(sd_accuracy, 2)
    mean_specificity <- round(mean_specificity, 2)
    sd_specificity <- round(sd_specificity, 2)
    mean_f1score <- round(mean_f1score, 2)
    sd_f1score <- round(sd_f1score, 2)
    # Update the classificationResults
    classificationResults(
      paste(
        "XGBoost Classification Results (Average of", k, "folds):",
        "\nSensitivity: ", mean_sensitivity, " ±", sd_sensitivity, "",
        "\nSpecificity: ", mean_specificity, " ±", sd_specificity, "",
        "\nAccuracy: ", mean_accuracy, " (±", sd_accuracy, ")",
        "\nF1 Score: ", mean_f1score, " ±", sd_f1score, ""
      )
    )
    
    # Update the classification results in the GUI
    output$sensitivityOutput <- renderText({
      paste("Sensitivity: ", mean_sensitivity, " ±", sd_sensitivity, "")
    })
    
    output$accuracyOutput <- renderText({
      paste("Accuracy: ", mean_accuracy, " ±", sd_accuracy, "")
    })
    
    output$specificityOutput <- renderText({
      paste("Specificity: ", mean_specificity, " ±", sd_specificity, "")
    })
    
    output$f1scoreOutput <- renderText({
      paste("F1 Score: ", mean_f1score, " ±", sd_f1score, "")
    })
    
    
    X$Label <- factor(X$Label, levels = c(all_labels))
    
    # Results <- data.frame(
    #   #Iteration = rep(1:5, each = 5),
    #   Class = rep(levels(X$Label), times = iterations * 5*NG),
    #   Sensitivity = c(Sen[, 1], Sen[, 2],Sen[, 3]),
    #   Specificity = c(Spe[, 1], Spe[, 2],Spe[, 3] ),
    #   Accuracy = c(Acc[, 1], Acc[, 2],Acc[, 3]),
    #   F1Score = c(F1[, 1], F1[, 2],F1[, 3])
    # )
    # 
    # Results <- Results[!is.na(Results$Class), ]
    
    
    # Function to scale values between 0 and 1
    normalize_values <- function(x) {
      (x - min(x, na.rm = TRUE)) / diff(range(x, na.rm = TRUE))
    }
    
    # 
    output$boxplotOutput <- renderPlot({
      # Access the Outcome object from input
      if(NG>2){
        if (!is.null(Results)) {
          # Set up a larger plotting area
          par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
          
          # Define color mapping for each class
          class_colors <- c("CIR" = "yellow", "CON" = "green","LN" = "orange","LX" = "blue", "HCC" = "red")
          
          # Function to create a boxplot with specified y-axis label
          create_boxplot <- function(data, y_label) {
            ggplot(data, aes(x = Class, y = normalize_values(.data[[y_label]]), fill = Class)) +
              geom_boxplot() +
              labs(title = "", y = y_label) +
              ylim(0, 1) +
              scale_fill_manual(values = class_colors) +
              theme(
                axis.text = element_text(face = "bold"),  # Bold axis text
                axis.title = element_text(face = "bold"),  # Bold axis titles
                legend.title = element_text(face = "bold"),  # Bold legend title
                legend.text = element_text(face = "bold"),  # Bold legend text
                panel.background = element_rect(fill = "white", color = "black", size = 1),
                plot.margin = margin(10, 10, 10, 10),
                legend.margin = margin(0, 0, 0, 0),
                legend.key = element_blank()  # Remove the legend key background
              )
          }
          
          # Create boxplots for each performance metric
          p1 <- create_boxplot(Results, "Specificity")
          p2 <- create_boxplot(Results, "Accuracy")
          p3 <- create_boxplot(Results, "Sensitivity")
          p4 <- create_boxplot(Results, "F1Score")
          
          # Combine the plots
          multiplot <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2) +
            ggtitle("") +
            theme(plot.background = element_rect(fill = "white", color = "black", size = 1))
          print(multiplot)
        }
        
      }else{
        
        if (!is.null(Outcome)) {
          boxplot(
            Outcome$Sensitivity,
            Outcome$Specificity,
            Outcome$Accuracy,
            Outcome$F1Score,
            col = rainbow(length(Outcome)),
            names = c("Sensitivity", "Specificity", "Accuracy", "F1-Score"),
            main = paste("XGBoost:",allCombinedNames()),
            
            ylim = c(0, 1)
          )
        }
      }
    })
    
    #   
    #   output$downloadPlot <- downloadHandler(
    #     filename = function() {
    #       
    #       paste("model_evaluation", Sys.Date(), ".jpg", sep = "_")
    #     },
    #     content = function(file) {
    #       
    #       # Generate the Model Evaluation plot
    #       par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
    #       
    #       # Define color mapping for each class
    #       class_colors <- c("CIR" = "yellow", "CON" = "green", "HCC" = "red")
    #       
    #       # Function to create a boxplot with specified y-axis label
    #       create_boxplot <- function(data, y_label) {
    #         ggplot(data, aes(x = Class, y = normalize_values(.data[[y_label]]), fill = Class)) +
    #           geom_boxplot() +
    #           labs(title = "Model Performance", y = y_label) +
    #           ylim(0, 1) +
    #           scale_fill_manual(values = class_colors) +
    #           theme(
    #             axis.text = element_text(face = "bold"),  # Bold axis text
    #             axis.title = element_text(face = "bold"),  # Bold axis titles
    #             legend.title = element_text(face = "bold"),  # Bold legend title
    #             legend.text = element_text(face = "bold"),  # Bold legend text
    #             panel.background = element_rect(fill = "white", color = "black", size = 1),
    #             plot.margin = margin(10, 10, 10, 10),
    #             legend.margin = margin(0, 0, 0, 0),
    #             legend.key = element_blank()  # Remove the legend key background
    #           )
    #       }
    #       
    #       # Create boxplots for each performance metric
    #       p1 <- create_boxplot(Results, "Specificity")
    #       p2 <- create_boxplot(Results, "Accuracy")
    #       p3 <- create_boxplot(Results, "Sensitivity")
    #       p4 <- create_boxplot(Results, "F1Score")
    #       
    #       # Combine the plots
    #       multiplot <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2) +
    #         ggtitle("Model Performance") +
    #         theme(plot.background = element_rect(fill = "white", color = "black", size = 1))
    #       print(multiplot)
    #       
    #       # Save the plot as a JPG file using jpeg()
    #       dev.copy(jpeg, file)
    #       dev.off()
    #     }
    #   )
    #   
    #   
    #   classificationResults(message) 
    # })
    
    
    output$downloadPlot <- downloadHandler(
      filename = function() {
        
        paste("model_evaluation", Sys.Date(), ".jpg", sep = "_")
      },
      content = function(file) {
        
        # Generate the Model Evaluation plot
        boxplot(
          Outcome$Sensitivity,
          Outcome$Specificity,
          Outcome$Accuracy,
          Outcome$F1Score,
          col = rainbow(length(Outcome)),
          names = c("Sensitivity", "Specificity", "Accuracy", "F1-Score"),
          main = allCombinedNames(),
          
          ylim = c(0, 1)
        )
        
        # Save the plot as a JPG file using jpeg()
        dev.copy(jpeg, file)
        dev.off()
      }
    )
    
    
    classificationResults(message) 
  })
  
  
  observeEvent(input$BatchButton, {
    req(cleanedDatasetList())
    
    method <- input$BatchMethod
    
    if (method == "ComBat") {
      req(selectedDatasets(), selectedFeaturesOutput())
      
      if (!is.null(selectedDatasets())) {
        # Get the selected labels from the checkbox inputs
        leftLabels <- input$leftLabelSelection
        rightLabels <- input$rightLabelSelection
        # Combine the selected labels
        selectedLabels <- c(leftLabels, rightLabels)
        P <- selectedDatasets()
        # Filter the dataset based on the selected labels
        filteredDataset <- lapply(P, function(sublist) {
          if ("Label" %in% names(sublist)) {
            subset <- sublist[sublist$Label %in% selectedLabels, ]
          } else {
            subset <- sublist[sublist[, 1] %in% selectedLabels, , drop = FALSE]
          }
          t(subset)
        })
        
        X1 <- filteredDataset[[1]]
        X1=t(as.data.frame(X1))
        X2 <- filteredDataset[[2]]
        X2=t(as.data.frame(X2))
        common_cols2 <- intersect(colnames(X1[,-1]), colnames(X2[,-1]))
        
        
        if (!is.data.frame(X1)) {
          X1 <- as.data.frame(X1)
        }
        
        X1 <-  X1[X1$Label == leftLabels, ]
        L1=data.frame(Label=X1$Label)
        X1_Valid <- X1[, intersect(colnames(X1),common_cols2)]
        X1_Valid <- as.data.frame( X1_Valid)
        X1_Valid <- sapply( X1_Valid, as.numeric)
        X1_Valid <- as.matrix( X1_Valid)
        
        if (!is.data.frame(X2)) {
          X2 <- as.data.frame(X2)
        }
        
        X2 <-  X2[X2$Label == rightLabels, ]
        L2=data.frame(Label=X2$Label)
        X2_Valid <- X2[, intersect(colnames(X2),common_cols2)]
        X2_Valid <- as.data.frame( X2_Valid)
        X2_Valid <- sapply( X2_Valid, as.numeric)
        X2_Valid <- as.matrix( X2_Valid)
        common_cols <- intersect(colnames(X1_Valid), colnames(X2_Valid))
        X1_Valid <- X1_Valid[, common_cols]
        X2_Valid <- X2_Valid[, common_cols]
        X_Valid=rbind(X1_Valid,X2_Valid)
        LL=rbind(L1,L2)
        y_valid <- LL
        y_valid$Label <- ifelse(y_valid$Label %in% leftLabels, 0, ifelse(y_valid$Label %in% rightLabels, 1, NA))
        y_valid <- unlist(y_valid$Label)
        y_valid <- as.numeric(y_valid)
        dataset=cbind(LL,X_Valid)
        Label <- dataset[, 1]
        labels <- as.factor(Label)
        features <- dataset[, -1]
        features <- sapply(features, as.numeric)
        combat_result <- suppressMessages(ComBat(dat = as.matrix(t(features)), batch = labels,par.prior = TRUE))
        combat_result= t(combat_result)
        # Combine the corrected features with the batch variable
        combat_result=combat_result[,intersect(colnames(combat_result), selectedFeaturesOutput())]
        combat_result=data.frame(combat_result)
        combat_result <- cbind(Label, combat_result)
        XX_Valid=combat_result
        
        if (input$oversampleCheckbox) {
          
          XX_Valid[, -1] <- sapply(XX_Valid[, -1], as.numeric)
          desired_order <- c( leftLabels,rightLabels)  # Specify the desired order of labels
          XX_Valid$Label <- factor(XX_Valid$Label,levels = desired_order)
          class_counts <- table(XX_Valid$Label)
          desired_samples <- max(class_counts)
          L_data <- subset(XX_Valid, Label == leftLabels)
          R_data <- subset(XX_Valid, Label == rightLabels)
          oversampled_R_data <- SMOTE(Label ~ ., XX_Valid, perc.over=100, perc.under=0, k=5)
          oversampled_data <- rbind( L_data,oversampled_R_data)
          oversampled_data <- oversampled_data[sample(nrow(oversampled_data)), ]
          XX_Valid=oversampled_data
          ordering_var <- match(XX_Valid$Label, desired_order)
          
          # Sort the dataframe based on the ordering variable
          combat_result <- XX_Valid[order(ordering_var), ]  
        }
        
        cleanedDatasetList(combat_result)
        selectedDatasets_V(cleanedDatasetList())
        processText("Batch Effect Correction")
      }
      
    } else {
      req(selectedDatasets(), selectedFeaturesOutput())
      
      if (!is.null(selectedDatasets())) {
        # Get the selected labels from the checkbox inputs
        leftLabels <- input$leftLabelSelection
        rightLabels <- input$rightLabelSelection
        # Combine the selected labels
        selectedLabels <- c(leftLabels, rightLabels)
        P <- selectedDatasets()
        # Filter the dataset based on the selected labels
        filteredDataset <- lapply(P, function(sublist) {
          if ("Label" %in% names(sublist)) {
            subset <- sublist[sublist$Label %in% selectedLabels, ]
          } else {
            subset <- sublist[sublist[, 1] %in% selectedLabels, , drop = FALSE]
          }
          t(subset)
        })
        
        X1 <- filteredDataset[[1]]
        X1=t(as.data.frame(X1))
        X2 <- filteredDataset[[2]]
        X2=t(as.data.frame(X2))
        common_cols2 <- intersect(colnames(X1[,-1]), colnames(X2[,-1]))
        
        if (!is.data.frame(X1)) {
          X1 <- as.data.frame(X1)
        }
        
        X1 <-  X1[X1$Label == leftLabels, ]
        L1=data.frame(Label=X1$Label)
        X1_Valid <- X1[, intersect(colnames(X1),common_cols2)]
        X1_Valid <- as.data.frame( X1_Valid)
        X1_Valid <- sapply( X1_Valid, as.numeric)
        X1_Valid <- as.matrix( X1_Valid)
        
        if (!is.data.frame(X2)) {
          X2 <- as.data.frame(X2)
        }
        
        X2 <-  X2[X2$Label == rightLabels, ]
        L2=data.frame(Label=X2$Label)
        X2_Valid <- X2[, intersect(colnames(X2),common_cols2)]
        X2_Valid <- as.data.frame( X2_Valid)
        X2_Valid <- sapply( X2_Valid, as.numeric)
        X2_Valid <- as.matrix( X2_Valid)
        common_cols <- intersect(colnames(X1_Valid), colnames(X2_Valid))
        X1_Valid <- X1_Valid[, common_cols]
        X2_Valid <- X2_Valid[, common_cols]
        X_Valid=rbind(X1_Valid,X2_Valid)
        LL=rbind(L1,L2)
        features=X_Valid
        features <- cbind(LL,features)
        
        if (input$oversampleCheckbox) {
          XX_Valid=cbind(LL,X_Valid)
          desired_order <- c( leftLabels,rightLabels)  # Specify the desired order of labels
          XX_Valid$Label <- factor(XX_Valid$Label, levels = desired_order)
          class_counts <- table(XX_Valid$Label)
          desired_samples <- max(class_counts)
          L_data <- subset(XX_Valid, Label == leftLabels)
          R_data <- subset(XX_Valid, Label == rightLabels)
          oversampled_R_data <- SMOTE(Label ~ ., XX_Valid, perc.over=100, perc.under=0, k=5)
          oversampled_data <- rbind(L_data,oversampled_R_data)
          oversampled_data <- oversampled_data[sample(nrow(oversampled_data)), ]
          XX_Valid=oversampled_data
          ordering_var <- match(XX_Valid$Label, desired_order)
          # Sort the dataframe based on the ordering variable
          features <- XX_Valid[order(ordering_var), ]
          
        }
        
        cleanedDatasetList(features)
        selectedDatasets_V(cleanedDatasetList())
        processText("Data Not ComBated")
      }
    }
    
  })
  
  
  
  observeEvent(input$classificationButton_v, {
    req(selectedDatasets_V(), selectedFeaturesOutput())
    
    # Perform classification based on the selected method
    if (input$classificationMethod_v == "Trained Model") {
      message <- "Random Forest Classification Method"
      
      if (!is.null(selectedDatasets_V())) {
        # Get the selected labels from the checkbox inputs
        leftLabels <- input$leftLabelSelection
        rightLabels <- input$rightLabelSelection
        # Combine the selected labels
        selectedLabels <- c(leftLabels, rightLabels)
        P <- selectedDatasets_V()
        X_Valid=P[,-1]
        y_valid <- data.frame(Label=P[,1])
        y_valid$Label <- ifelse(y_valid$Label %in% leftLabels, 0, ifelse(y_valid$Label %in% rightLabels, 1, NA))
        y_valid <- unlist(y_valid$Label)
        y_valid <- as.numeric(y_valid)
        model <- classificationModel()
        # Subset the validation dataset to include only the selected features
        cm=list()
        Outcome=data.frame()
        X_Valid=as.data.frame(X_Valid)
        X_Valid<- sapply(X_Valid, as.numeric)
        X_Valid= as.matrix(X_Valid)
        for (j in 1:input$numIterationsC)
        {
          feature_names <- model[[j]]$feature_names
          feature_order <- match(feature_names, colnames(X_Valid))
          X_Valid <- X_Valid[, feature_order]
          pred <- predict(model[[j]], X_Valid)
          pred <- ifelse(pred > 0.5, 1, 0)
          # Evaluate the predictions
          cm[[j]] <- caret::confusionMatrix(
            factor(y_valid, levels = c(0, 1)),
            factor(pred, levels = c(0, 1))
          )
          # Extract evaluation metrics
          Outcome[j, 1:4] <- data.frame(
            specificity = cm[[j]][["byClass"]][["Specificity"]],
            sensitivity = cm[[j]][["byClass"]][["Sensitivity"]],
            accuracy = cm[[j]][["overall"]][["Accuracy"]],
            f1score = cm[[j]][["byClass"]][["F1"]]
          )
          
        }
        # Calculate the mean and standard deviation of evaluation metrics
        mean_sensitivity <- mean(Outcome$sensitivity,na.rm = TRUE) * 100
        sd_sensitivity <- sd(Outcome$sensitivity,na.rm = TRUE) * 100
        mean_accuracy <- mean(Outcome$accuracy,na.rm = TRUE) * 100
        sd_accuracy <- sd(Outcome$accuracy,na.rm = TRUE) * 100
        mean_specificity <- mean(Outcome$specificity,na.rm = TRUE) * 100
        sd_specificity <- sd(Outcome$specificity,na.rm = TRUE) * 100
        mean_f1score <- mean(Outcome$f1score,na.rm = TRUE) * 100
        sd_f1score <- sd(Outcome$f1score,na.rm = TRUE) * 100
        
        # Format the values to two decimal places
        mean_sensitivity <- round(mean_sensitivity, 2)
        sd_sensitivity <- round(sd_sensitivity, 2)
        mean_accuracy <- round(mean_accuracy, 2)
        sd_accuracy <- round(sd_accuracy, 2)
        mean_specificity <- round(mean_specificity, 2)
        sd_specificity <- round(sd_specificity, 2)
        mean_f1score <- round(mean_f1score, 2)
        sd_f1score <- round(sd_f1score, 2)
        
        # Update the classificationResults
        classificationResults(
          paste(
            "XGBoost Classification Results (Average of", 5, "folds):",
            "\nSensitivity: ", mean_sensitivity, " ±", sd_sensitivity, "",
            "\nSpecificity: ", mean_specificity, " ±", sd_specificity, "",
            "\nAccuracy: ", mean_accuracy, " (±", sd_accuracy, ")",
            "\nF1 Score: ", mean_f1score, " ±", sd_f1score, ""
          )
        )
        
        # Update the classification results in the GUI
        output$sensitivityOutput_v <- renderText({
          paste("Sensitivity: ", mean_sensitivity, " ±", sd_sensitivity, "")
        })
        
        output$accuracyOutput_v <- renderText({
          paste("Accuracy: ", mean_accuracy, " ±", sd_accuracy, "")
        })
        
        output$specificityOutput_v <- renderText({
          paste("Specificity: ", mean_specificity, " ±", sd_specificity, "")
        })
        
        output$f1scoreOutput_v <- renderText({
          paste("F1 Score: ", mean_f1score, " ±", sd_f1score, "")
        })
        
        
        output$ModelEvaluation <- renderPlot({
          # Access the Outcome object from input
          if (!is.null(Outcome)) {
            boxplot(
              Outcome$sensitivity,
              Outcome$specificity,
              Outcome$accuracy,
              Outcome$f1score,
              col = rainbow(length(Outcome)),
              names = c("Sensitivity", "Specificity", "Accuracy", "F1-Score"),
              main =  allCombinedNames(), ylim = c(0, 1)
            )
          }
        })
        
      }
      
      
      
      output$ModelEvaluation <- renderPlot({
        # Access the Outcome object from input
        if (!is.null(Outcome)) {
          boxplot(
            Outcome$sensitivity,
            Outcome$specificity,
            Outcome$accuracy,
            Outcome$f1score,
            col = rainbow(length(Outcome)),
            names = c("Sensitivity", "Specificity", "Accuracy", "F1-Score"),
            main =  allCombinedNames(), ylim = c(0, 1)
          )
        }
      })
      
      
      
      output$downloadPlot_v <- downloadHandler(
        filename = function() {
          paste("model_evaluation", Sys.Date(), ".jpg", sep = "_")
        },
        content = function(file) {
          # Generate the Model Evaluation plot
          boxplot(
            Outcome$sensitivity,
            Outcome$specificity,
            Outcome$accuracy,
            Outcome$f1score,
            col = rainbow(length(Outcome)),
            names = c("Sensitivity", "Specificity", "Accuracy", "F1-Score"),
            main = allCombinedNames(),
            ylim = c(0, 1)
          )
          # Save the plot as a JPG file using jpeg()
          dev.copy(jpeg, file)
          dev.off()
        }
      )
      
      
    } else {
      # Handle other classification methods
      message <- "Unknown Classification Method"
    }
    classificationResults(message)
  })
  
  
  
  observeEvent(input$VenDiagramButton, {
    req(selectedDatasets())
    # Get the selected labels from the checkbox inputs
    leftLabels <- input$leftLabelSelection
    rightLabels <- input$rightLabelSelection
    # Combine the selected labels
    selectedLabels <- c(leftLabels, rightLabels)
    P <- selectedDatasets()
    # Filter the dataset based on the selected labels
    filteredDataset <- lapply(P, function(sublist) {
      if ("Label" %in% names(sublist)) {
        subset <- sublist[sublist$Label %in% selectedLabels, ]
      } else {
        subset <- sublist[sublist[, 1] %in% selectedLabels, , drop = FALSE]
      }
      (subset)
    })
    L <- length(selectedDatasets())
    a <- list()
    b <- data.frame()
    
    for (i in 1:L) {
      dataset <- filteredDataset[[i]]
      a[[i]] <- colnames(dataset[, -1])
      b[1, i] <- ncol(dataset) - 1
    }
    max_length <- max(b)
    c <- matrix(NA, nrow = max_length, ncol = L)
    
    for (i in 1:L) {
      q <- paste("a", i, sep = "")
      s <- a[[i]]
      s <- c(s, rep(NA, max_length - length(s)))  # Pad vector s with NA values
      c[, i] <- s
    }
    
    c <- as.data.frame(c)
    fill_colors <- c("thistle", rep("gray", L - 1))
    category.names <- c("Original", paste0("Validation ", 1:(L-1)))
    venn.plot <- ggVennDiagram(
      x = as.list(c),
      category.names = category.names,
      fill = fill_colors
    )
    
    # Display the Venn diagram in the GUI
    output$VenDiagram <- renderPlot({
      plot(venn.plot)
    })
    
    # Handler for downloading the Venn diagram
    output$downloadVenDiagram <- downloadHandler(
      filename = function() {
        paste("venn_diagram", Sys.Date(), ".jpg", sep = "_")
        
      },
      content = function(file) {
        # Generate the Venn diagram
        venn.plot <- ggVennDiagram(
          x = as.list(c),
          category.names = category.names,
          fill = fill_colors
        )
        # Save the Venn diagram as a PNG file
        ggsave(file, venn.plot)
      }
    )
  })
  
}
shinyApp(ui, server)