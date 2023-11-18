import funzioni as f
import numpy as np

def main():
    TrainXpath="EmberXTrain.csv"
    TrainYpath="EmberYTrain.csv"
    print("Carico in memoria il file "+TrainXpath)
    X=f.load(TrainXpath)
    print("Carico in memoria il file "+TrainYpath)
    Y=f.load(TrainYpath)

    file_descrizione_x = 'descrizione_X.csv'
    file_descrizione_y = 'descrizione_Y.csv'
    print("Sto salvando le statistiche di ogni attributo del dataset X nel file "+file_descrizione_x)   
    f.preElaborationData(X, file_descrizione_x)
    print("Sto salvando le statistiche di ogni attributo del dataset Y nel file "+file_descrizione_y)   
    f.preElaborationData(Y, file_descrizione_y)

    print("Sto creando un boxplot per ciascun attributo elencato nella lista raggrupati per classe.")  
    print("Le immagini vengono salvate sotto la cartella /boxplot.")  
    f.preBoxPlotAnalysisData(X, Y)

    file_mutual_info_rank="mutual_info_rank.csv"
    seed=42
    np.random.seed(seed)
    print("Sto calcolando la mutualInfoClassif sulla lista delle variabili indipendenti.")
    rank=f.mutualInfoRank(X,Y,seed,file_mutual_info_rank)

    print("Sto selezionando le feature con Mutual Info maggiore di 0.1.")
    selectedfeatures = f.topFeatureSelect(rank,0.1)
    XSelected=X.loc[:, selectedfeatures]

    print("Sto calcolando la PCA di X, recuperando la lista dei nomi e la lista delle explained variance.")
    pca,pcalist,explained_variance=f.pca(X)

    print("Sto trasformando il dataframe usando la PCA.")
    XPCA=f.applyPCA(X,pca,pcalist.tolist())

    np.savetxt("explained_variance.csv", explained_variance, delimiter=',')
    print("Sto selezionando le PC che presentano una somma della varianza maggiore di 0.99.")
    n=f.numberOfTopPCSelect(explained_variance,0.99)
    print("Ho selezionato :"+str(n)+" PC.")
    XPCASelected=XPCA.iloc[:,1:(n+1)]
    
if __name__ == "__main__":
    main()