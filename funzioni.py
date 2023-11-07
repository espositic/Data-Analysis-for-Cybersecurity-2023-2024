import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm


def load(path):
    """
        Carica i dati da un file CSV in base al path indicato.
        :param path: percorso del CSV da caricare
        :return: dataframe contenente i dati
    """
    return pd.read_csv(path)


def preElaborationData(data, filename):
    """
        Fornisce le statistiche degli attributi del dataset
        I valori min,max, avg ecc sono salvati su un file csv
        :param data: dataset
        :param filename: nome del file csv in cui salvare i dati
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
    :param X: Dataframe X
    :param Y: Dataframe Y
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
    :param X: Dataframe X
    :param Y: Dataframe Y
    :return: lista di tuple (coppie nome/valore) ordinate in modo decrescente per mutual info
    """
    independentList = list(X.columns.values)
    mutual_info_values = []
    
    for column in tqdm(independentList, desc="Calculating Mutual Info"):
        mi = mutual_info_classif(X[[column]], np.ravel(Y), discrete_features=False, random_state=seed)
        mutual_info_values.append(mi[0])

    results = list(zip(independentList, mutual_info_values))
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Creazione di un DataFrame per i risultati
    result_df = pd.DataFrame(results, columns=['Nome', 'Valore'])
    
    # Salvataggio in un file CSV
    result_df.to_csv(output_file, index=False)
