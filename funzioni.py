import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from tqdm import tqdm


def load(path):
    """
    Carica i dati da un file CSV in base al path indicato.
    """
    return pd.read_csv(path)


def preElaborationData(data, filename):
    """
    Fornisce le statistiche degli attributi del dataset
    I valori min,max, avg ecc sono salvati su un file csv
    """
    features_list = data.columns
    values = []
    for feature in features_list:
        values.append(data[feature].describe())
    statistics = pd.DataFrame(values)
    statistics.to_csv(filename, sep=";")


def preBoxPlotAnalysisData(X, Y):
    """
    Crea un box plot per ciascun attributo elencato nella lista raggrupati per classe
    Il box plot è salvato come immagine nella cartella boxplot
    """
    output_directory = 'boxplot'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    num_columns = len(X.columns)
    for i, column in enumerate(tqdm(X.columns, desc="Progress")):
        data = pd.DataFrame(X[column])
        data['Label'] = Y['Label']

        fig, ax = plt.subplots()
        data.boxplot(by='Label', ax=ax)
        ax.set_title(f'Boxplot for {column}')

        output_path = os.path.join(output_directory, f'boxplot_{column}.png')
        plt.savefig(output_path)
        plt.close()


def mutualInfoRank(X,Y,seed,output_file):
    """
    Calcola la mutualInfoClassif sulla lista delle variabili indipendenti
    """
    independentList = list(X.columns.values)
    mutual_info_values = []
    
    for column in tqdm(independentList, desc="Calculating Mutual Info"):
        mi = mutual_info_classif(X[[column]], np.ravel(Y), discrete_features=False, random_state=seed)
        mutual_info_values.append(mi[0])

    results = list(zip(independentList, mutual_info_values))
    results.sort(key=lambda x: x[1], reverse=True)
    result_df = pd.DataFrame(results, columns=['Nome', 'Valore'])
    result_df.to_csv(output_file, index=False)
    return result_df

def topFeatureSelect(features,threshold):
    '''
    Seleziona le faeature con MI >= threshold
    '''
    selected_features = features[features['Valore'] >= threshold]
    return selected_features['Nome'].tolist()

def pca(dataframe):
    '''
    Calcola la PCA di X e restituisce la lista dei nomi e la lista delle explained variance delle pc 
    '''
    pca_model = PCA()
    return pca_model.fit(dataframe), pca_model.components_, pca_model.explained_variance_ratio_

def applyPCA(input_df, pca_model, pc_names):
    '''
    Applica la PCA al Dataframe
    '''
    transformed_data = pca_model.transform(input_df)
    pc_df = pd.DataFrame(transformed_data, columns=pc_names)
    return pc_df

def numberOfTopPCSelect(explained_variance_ratio, threshold):
    '''
    Restituisce il numero delle prime PC le cui explained variance sommata è maggiore a threshold
    '''
    sum=0
    i=0
    while sum < threshold and i < len(explained_variance_ratio):
        sum += explained_variance_ratio[i]
        i += 1
    return i