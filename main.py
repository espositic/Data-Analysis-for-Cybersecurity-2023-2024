from funzioni import load, preElaborationData, preBoxPlotAnalysisData, mutualInfoRank
import numpy as np

def main():
    TrainXpath="EmberXTrain.csv"
    TrainYpath="EmberYTrain.csv"
    print("Carico in memoria il file "+TrainXpath)
    X=load(TrainXpath)
    print("Carico in memoria il file "+TrainYpath)
    Y=load(TrainYpath)

    file_descrizione_x = 'descrizione_X.csv'
    file_descrizione_y = 'descrizione_Y.csv'
    print("Sto salvando le statistiche di ogni attributo del dataset X nel file "+file_descrizione_x)   
    preElaborationData(X, file_descrizione_x)
    print("Sto salvando le statistiche di ogni attributo del dataset Y nel file "+file_descrizione_y)   
    preElaborationData(Y, file_descrizione_y)

    print("Sto creando un boxplot per ciascun attributo elencato nella lista raggrupati per classe.")  
    print("Le immagini vengono salvate sotto la cartella /boxplot.")  
    preBoxPlotAnalysisData(X, Y)

    file_mutual_info_rank="mutual_info_rank.csv"
    seed=42
    np.random.seed(seed)
    print("Sto calcolando la mutualInfoClassif sulla lista delle variabili indipendenti.")
    rank=mutualInfoRank(X,Y,seed,file_mutual_info_rank)
    print("Calcolo completato.")

if __name__ == "__main__":
    main()