## Classifier

Classification of incoming flows using machine learning

The project is based upon the master thesis "[Network Monitoring and Attack Detection](http://www.tik.ee.ethz.ch/file/bfc340c7a832cb8696a98a3e5504a0b0/task_description_attack-detection.pdf)".
Reusing a lot of their approach and code, a model can be trained up against labelled network flows from the lockedshield exercises 2017 and 2018.
Unlike in the master thesis (which uses FlowMeter to extract features), we use Tranalyzer for feature extraction.
Once the most significant features for prediction are identified, a model can be trained with said feature set.

### General Procedure
#### Usage
1. Parse network traffic to network flows
1. Use hex_split.py to expand hex features
1. Call classifier.py with a trained model to predict data
1. Process output (malicious flows) of classifier.py further

#### Training
1. Have a network traffic file, parse it to flows while extracting features
1. Using csv_labeller.py label the flows if they are malicious or not (based on red team IP addresses from the last years) 
1. Using hex_split.py expand the hex features
1. Create needed arrays in ml_data containing all headers, effective features (used for classifying), categorical features (used to remove these since we have numerical feature constraints)
1. Generate train and test sets (see ml_supervised.py#DATASET GENERATION)
1. Analyse importance of features and create corresponding csv files (see ./data/rfe_removed_features_list-Tranalyzer-full-numerical.txt) (see ml_supervised.py#FEATURE SELECTION)
    1. NOTE: You may need to remove categorical features from the comma separated list since they cannot be processed.
    1. Afterwards create in ml_data a featureset_ordered_by_importance for convenience (create array of string out of generated csv file and reverse it)
1. Verify with how many features we get the best results (call to ml_feature_selection.train_and_test_with_different_feature_subsets)
1. Train the model with the amount of features (ordered by importance) and save it
1. Verify performance/accuracy of the model
1. Once happy with it, adapt classify.py to extract the same amount and same features (see classify.py#get_feature_indices)

Afterwards classify.py is ready to be used for classification

### Open Tasks
* (Fixing) Monitoring mode: In a real time scenario incoming flags need to be captured, passed to tranalyzer and to hex_split.py for expansion of hex features and finally processed by classifier.py for classification. Tranalyzer will be running in monitoring mode capturing any incoming traffic and extracting flows and features. However hex_split.py and classifier.py are currently implemented to read everything in standard input and will finish once everything in standard input is processed. 
* Find featureset
** Naively using the full feature set only got us so far. With the initial tranalyzer feature set (removing all non-numerical categories and therefore also hex encoded ones) we didn't get statisfactory results. Now after the introduction of hex_split.py the flows files have been regenerated including train/test sets for k-fold cross validation. However we weren't yet able to analyse feature importance since now with the expanded features and increased raw file size, the memory consumption of the feature analysis taken over from the master thesis exceeded the available memory
