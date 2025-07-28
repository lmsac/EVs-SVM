# EVs-SVM
"Machine Learning-Driven Extracellular Vesicles Peptidomics Powers Precision Classification of Endometrial Cancer Molecular Subtypes"


This repository includes original data, model selection, model construction and validation for endometrial cancer screening and molecular subtyping.

Repository Structure


Diagnosis data: Store the original training set, test set, and validation set of the screening model.


Molecular subtype data: Store the original training and testing data for the molecular typing multi-classification model


Identification: Peptides identified by peptidome via LC-MS and Peptides matching 29 MALDI-TOF MS key features within 2000 ppm tolerance


Subtype result: Store the selection results of molecular subtyping models, as well as the results of hierarchical binary classification and multi-classification models


Sites where MALDI-TOF MS profiles were collected: University of Fudan, China


For details on the dataset extraction and preprocessing, please refer to the Methods section in the article corresponding to the publication https://www.nature.com/articles/s41591-021-01619-9.
Preprocessing of the spectra was conducted using the R packages MALDIquant and MALDIquantForeign. This included square root transformation, Savitzky-Golay smoothing, and baseline correction to improve signal quality. Peak detection was performed using a signal-to-noise ratio (S/N) threshold of 6, a half-window size of 10, and a tolerance of 0.005. The detected m/z values were aligned across the mass spectra to create a feature peak matrix, which contained the m/z values of peaks and their corresponding signal intensity information from each individual data.
