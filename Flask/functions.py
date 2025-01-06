import pandas as pd
import sklearn.cluster._hdbscan
import sklearn.metrics
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats




# This is the index dictionary which stores where each data category is located within the main dataframe
df_index_dict = {'Accommodation type (5 categories)': [2, 7], 'Adults and children in household (11 categories)': [7, 17], 'Car or van availability (3 categories)': [17, 19], 'Accommodation by type of dwelling (9 categories)': [19, 28], 'Combination of ethnic groups in household (8 categories)': [28, 35], 'Combination of religions in household (15 categories)': [35, 49], 'Dependent children in household and their age - indicator (3 categories)': [49, 51], 'Household Reference Person previously served in UK armed forces (5 categories)': [51, 55], 'Household deprivation (6 categories)': [55, 60], 'Household deprived in the education dimension (3 categories)': [60, 62], 'Household deprived in the employment dimension (3 categories)': [62, 64], 'Household deprived in the health and disability dimension (3 categories)': [64, 66], 'Household deprived in the housing dimension (3 categories)': [66, 68], 'Household language (English and Welsh) (5 categories)': [68, 72], 'Household size (5 categories)': [72, 77], 'Household type (6 categories)': [77, 82], 'Households with students or schoolchildren living away during term-time (4 categories)': [82, 86], 'Lifestage of Household Reference Person(13 categories)': [86, 98], 'Multiple ethnic groups in household (6 categories)': [98, 103], 'Multiple main languages in household (3 categories)': [103, 105], 'Number of Bedrooms (5 categories)': [105, 109], 'Number of adults in employment in household (5 categories)': [109, 113], 'Number of adults in household (3 categories)': [113, 115], 'Number of disabled adults in household (4 categories)': [115, 118], 'Number of disabled people in household (4 categories)': [118, 121], 'Number of disabled people in household whose day-to-day activities are limited a little (4 categories)': [121, 124], 'Number of disabled people in household whose day-to-day activities are limited a lot (4 categories)': [124, 127], 'Number of families in household (7 categories)': [127, 134], 'Number of people in household who previously served in UK armed forces (3 categories)': [134, 136], 'Number of people in household with a long-term heath condition but are not disabled (4 categories)': [136, 139], 'Number of people in household with no long-term health condition (4 categories)': [139, 142], 'Number of people per bedroom in household (5 categories)': [142, 146], 'Number of people per room in household (5 categories)': [146, 150], 'Number of people who work in household and their transport to work (18 categories)': [150, 167], 'Number of rooms (Valuation Office Agency) (6 categories)': [167, 173], 'Number of unpaid carers in household (6 categories)': [173, 178], 'Occupancy rating for bedrooms (5 categories)': [178, 182], 'Occupancy rating for rooms (5 categories)': [182, 186], 'Tenure of household (7 categories)': [186, 192], 'Type of central heating in household (13 categories)': [192, 204]}

# This imports the two main dataframes as global variables
main_df = pd.read_csv("static\main_df.csv")
main_df_no_norm = pd.read_csv("static\main_df_no_norm.csv")








def dataframeCreation(selection_list):
    """
    Creates a custom dataframe depending on the categories selected by the user

    Parameters
    ----------
        selection_list : list
            A list of category names which determine the contents of this new dataframe

    Returns
    -------
        cluster_df : DataFame
            A dataframe which contains the specified user selected categories for clustering

    """
    # After the user has selected the categories they want
    # A custom data frame is created by selecting the user attributes from the main df
    # cluster_df initially only has one column which contains all the LLA's
    cluster_df = pd.read_excel("static\LLA's.xlsx")
    for attribute in selection_list:
        column_idx = df_index_dict[attribute]
        cluster_df = pd.concat([cluster_df,main_df.iloc[:,column_idx[0]:column_idx[1]]],axis=1)
    cluster_df = cluster_df.loc[:,~cluster_df.columns.duplicated()].copy()
    return cluster_df

def dataframeCreationNoNorm(selection_list):
    """
    Creates a custom dataframe depending on the categories selected by the user without normalization

    Parameters
    ----------
        selection_list : list
            A list of category names which determine the contents of this new dataframe

    Returns
    -------
        cluster_df : DataFame
            A dataframe which contains the specified user selected categories for clustering

    """
    cluster_df = pd.read_excel("static\LLA's.xlsx")
    for attribute in selection_list:
        column_idx = df_index_dict[attribute]
        cluster_df = pd.concat([cluster_df,main_df_no_norm.iloc[:,column_idx[0]:column_idx[1]]],axis=1)
    # REFERENCE 
    # Gene (2016) Stack overflow, https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
    # Note this is to remove the duplicate column of the names of LLA's which is created when converting a dataframe into a html table
    cluster_df = cluster_df.loc[:,~cluster_df.columns.duplicated()].copy()
    return cluster_df


def preCluster(df:pd.DataFrame):
    """
    Performs final formatting of dataframe before it is clustered
    Removes LLA column from dataframe and then converts remaining values to a numpy array

    Parameters
    ----------
        df : DataFrame
            The dataframe to be clustered

    Returns
    -------
        local_auth_df : DataFrame
            A dataframe containing all the lower tier local authorities

        cluster_data_arr: np.array
            An numpy array containing all the values to be used in clustering
    """
    # We separate the names of the LLA's from the data as we can only have numbers for clustering
    # We store the names of the LLA's in a new df and remove them from the actual data with iloc
    # Additionally the data is converted to a numpy array as these work better with clustering in sklearn
    local_auth_df = df["Lower tier local authorities"]
    cluster_data_df = df.iloc[:,2:]
    cluster_data_arr = cluster_data_df.to_numpy()
    return local_auth_df, cluster_data_arr

def findBestCluster(silhouette_dict, labels_dict):
    """
    Finds the best clustering from each individual clustering method hyper parameter tuning

    Parameters
    ----------
        silhouette_dict : dict
            A dictionary where for each entry the key is the clustering method used and the values are their silhouette scores

        labels_dict : dict
            A dictionary where for each entry the key is the clustering method used and the values are their group labels
    """
    best_silhouette_key = max(silhouette_dict, key=silhouette_dict.get)
    label = labels_dict[best_silhouette_key]
    silhouette_scr = silhouette_dict[best_silhouette_key]
    return silhouette_scr, label


def kMeansClustering(cluster_data_arr):
    """
    Performs KMeans clustering on the data array

    Parameters
    ----------
        cluster_data_arr : np.array
            A numpy array containing all the values to be used in clustering

    Returns
    -------
        silhouette_scr : int
            The best silhouette score of the clustering
        
        label : list
            A list of int which represents the grouping of the clustering with the highest silhouette score

        "KMeans" : str
            A hardcoded string of the clustering method used
    """
    # We for loop over the clustering as we want to hyperparameter train the best number of clusters
    # The Kmeans algorithm is run 29 times to find what is the best number of clusters
    silhouette_dict = {}
    labels_dict = {}
    for no_clusters in range(1,30):
       model = sklearn.cluster.KMeans(n_clusters = no_clusters, init="k-means++")
       silhouetteErrorExcept(model,cluster_data_arr,silhouette_dict, no_clusters, labels_dict)
    # The best cluster function is called to find which hyperparameters gave the best clustering
    # The silhouette score, the cluster lables and the algorithm name are returned
    silhouette_scr, label = findBestCluster(silhouette_dict,labels_dict)
    return [silhouette_scr, label, "kMeans"]
    


def bisectingKmeansClustering( cluster_data_arr):
    """
    Performs Bisecting KMeans clustering on the data array

    Parameters
    ----------
        cluster_data_arr : np.array
            A numpy array containing all the values to be used in clustering

    Returns
    -------
        silhouette_scr : int
            The best silhouette score of the clustering
        
        label : list
            A list of int which represents the grouping of the clustering with the highest silhouette score

        "BisectingKmeans" : str
            A hardcoded string of the clustering method used
    """
    silhouette_dict = {}
    labels_dict = {}
    # We for loop over the clustering as we want to hyperparameter train the best number of clusters
    # The Bisecting Kmeans Algorithm is run 29 times to find what is the best number of clusters
    for no_clusters in range(1,30):
       model = sklearn.cluster.BisectingKMeans(n_clusters = no_clusters, init="k-means++")
       silhouetteErrorExcept(model,cluster_data_arr,silhouette_dict, no_clusters, labels_dict)
    # The best cluster function is called to find which hyperparameters gave the best clustering
    # The silhouette score, the cluster lables and the algorithm name are returned
    silhouette_scr, label = findBestCluster(silhouette_dict,labels_dict)
    return [silhouette_scr, label, "BisectingKmeans"]



def agglomerativeClustering( cluster_data_arr):
    """
    Performs Agglomerative clustering on the data array

    Parameters
    ----------
        cluster_data_arr : np.array
            A numpy array containing all the values to be used in clustering

    Returns
    -------
        silhouette_scr : int
            The best silhouette score of the clustering
        
        label : list
            A list of int which represents the grouping of the clustering with the highest silhouette score

        "AgglomerativeClustering" : str
            A hardcoded string of the clustering method used
    """
    silhouette_dict = {}
    labels_dict = {}
    # We for loop over the clustering as we want to hyperparameter train the best number of clusters
    # The agglomerative Algorithm is run 28 times to find what is the best number of clusters
    for no_clusters in range(2,30):
       model = sklearn.cluster.AgglomerativeClustering(n_clusters = no_clusters)
       silhouetteErrorExcept(model,cluster_data_arr,silhouette_dict, no_clusters, labels_dict)
    # The best cluster function is called to find which hyperparameters gave the best clustering
    # The silhouette score, the cluster lables and the algorithm name are returned
    silhouette_scr, label = findBestCluster(silhouette_dict,labels_dict)
    return [silhouette_scr, label, "AgglomerativeClustering"]

    
def dbscanClustering( cluster_data_arr):
    """
    Performs DBSCAN clustering on the data array

    Parameters
    ----------
        cluster_data_arr : np.array
            A numpy array containing all the values to be used in clustering

    Returns
    -------
        silhouette_scr : int
            The best silhouette score of the clustering
        
        label : list
            A list of int which represents the grouping of the clustering with the highest silhouette score

        "dbscanClustering" : str
            A hardcoded string of the clustering method used
    """
    silhouette_dict = {}
    labels_dict = {}
    # For DBSCAN clustering there is more than one hyperparameter to tune
    # Therefore we need to run two seperate loops to find the best ones
    # First of all we start with the eps_val which is the furthest distance between two points where
    # we still consider them to be neighbours to each other
    # When we find the best eps_val we pass that to the next loop
    # Where we hyperparameter train the no_min samples which is the minimum number of points within another points
    # neighbourhood 
    for eps_val in np.arange(0.1,0.5,0.1):
       model = sklearn.cluster.DBSCAN(eps=eps_val,min_samples=5)
       silhouetteErrorExceptDBCluster(model,cluster_data_arr,silhouette_dict, eps_val,labels_dict)
    best_eps = max(silhouette_dict, key=silhouette_dict.get)
    silhouette_dict = {}
    labels_dict = {}
    for no_min_samples in np.arange(2,11,1):
       model = sklearn.cluster.DBSCAN(eps=best_eps,min_samples=no_min_samples)
       silhouetteErrorExceptDBCluster(model,cluster_data_arr,silhouette_dict, no_min_samples, labels_dict)
    # The best cluster function is called to find which hyperparameters gave the best clustering
    # The silhouette score, the cluster lables and the algorithm name are returned
    silhouette_scr, label = findBestCluster(silhouette_dict,labels_dict)
    return [silhouette_scr, label, "dbscanClustering"]

def silhouetteErrorExcept(model, cluster_data_arr, silhouette_dict, indexer, labels_dict):
    """
    Runs the silhouette score function from sklearn by has a try except block to catch
    the error where a clustering method clusters all points together so a silhouette
    score cannot be computed

    Parameters
    ----------

    model : clustering model
        The model used in clustering

    cluster_data_arr : np.array
        A numpy array containing all the values to be used in clustering
    
    silhouette_dict : dict
        A dictionary where for each entry the key is the value of the hyperparameter used in the clustering and the values
        are the silhouette score the clustering achieved with that specific hyperparameter

    indexer : int
        The values of the hyperparameters being tuned

    labels_dict : dict
        A dictionary where for each entry the key is the value of the hyperparameter used in the clustering and the values
        are the group labels for all points from that clustering

    Returns
    -------
        None
    """
    # This try catch statement is used to catch a common error that can occur with silhouette scores and clustering
    # If the clustering algorithm only determine there to be one cluster the silhouette function will post an error
    # As silhouette score requires calculating points between different clusters if there is only one then this cannot happen
    # when the error is caught we replace the silhouette score with negative infinity
    # This was chosen as we don't want a value like 0 which in some cases could be the highest silhouette score
    labels = model.fit_predict(cluster_data_arr)
    try:
        silhouette_scr = sklearn.metrics.silhouette_score(cluster_data_arr,labels)
        silhouette_dict[indexer] = silhouette_scr
        labels_dict[indexer] = labels
    except ValueError:
        silhouette_dict[indexer] = float('-inf')
        labels_dict[indexer] = labels
    return


def silhouetteErrorExceptDBCluster(model, cluster_data_arr, silhouette_dict, indexer, labels_dict):
    """
    Same function as silhouetteErrorExcept, except this is made specifically for DBSCAN clustering.
    This is because DBSCAN creates outliers which are represented in the labels as -1. This will avoid the try except
    error catch as even if there is only one group in the clustering these outliers will make it seems like there is more than one.
    So this function checks there are more than one unique positive numbers in the labels of the clustering i.e. checking
    there is more that one cluster excluding outliers

    Parameters
    ----------

    model : clustering model
        The model used in clustering

    cluster_data_arr : np.array
        A numpy array containing all the values to be used in clustering
    
    silhouette_dict : dict
        A dictionary where for each entry the key is the value of the hyperparameter used in the clustering and the values
        are the silhouette score the clustering achieved with that specific hyperparameter

    indexer : int
        The values of the hyperparameters being tuned

    labels_dict : dict
        A dictionary where for each entry the key is the value of the hyperparameter used in the clustering and the values
        are the group labels for all points from that clustering

    Returns
    -------
        None
    """
    labels = model.fit_predict(cluster_data_arr)
    label_set = set()
    # We need this special silhouette function for BDSCAN as it can identify outliers
    # These are represented as -1 in the labels
    # This causes an issue as some cluster results ended with one real cluster and the other points becoming outliers
    # Becuase of this it lead to a very high silhouette score from improper clustering
    # Therefore this code make sure there is more than one real cluster ignoring outliers
    # If there is only one real cluster and the other points are outliers the silhouette score is set to negative
    for item in labels:
        if item >= 0:
            label_set.add(item)
    if len(label_set) <= 1:
        silhouette_dict[indexer] = float('-inf')
        labels_dict[indexer] = labels
    else:
        try:
            silhouette_scr = sklearn.metrics.silhouette_score(cluster_data_arr,labels)
            silhouette_dict[indexer] = silhouette_scr
            labels_dict[indexer] = labels
        except ValueError:
            silhouette_dict[indexer] = float('-inf')
            labels_dict[indexer] = labels
        return
        
def mainClusterLoop(cluster_df):
    """
    Loops through all the clustering methods

    Parameters
    ----------
        n/a

    Returns
    -------
    cluster_sil_dict : Dict
        A dictionary where for each entry the key is the clustering method used, and the values is the silhouette score

    cluster_info_dict : Dict
        A dictionary where for each entry the key is the clustering method used, and the values is the labels of the clustering
    """
    # This is the main clustering loop
    # Each algorithms best result is stored in two dictionaries
    # Y is equal to a list of [silhouette_score, labels, algorithm_name]
    cluster_info_dict = {}
    cluster_sil_dict = {}
    local, cluster_data_arr = preCluster(cluster_df)
    Y = kMeansClustering(cluster_data_arr)
    cluster_info_dict[Y[2]] = [Y[1]]
    cluster_sil_dict[Y[2]] = Y[0]
    Y = bisectingKmeansClustering(cluster_data_arr)
    cluster_info_dict[Y[2]] = [Y[1]]
    cluster_sil_dict[Y[2]] = Y[0]
    Y = agglomerativeClustering(cluster_data_arr)
    cluster_info_dict[Y[2]] = [Y[1]]
    cluster_sil_dict[Y[2]] = Y[0]
    Y = dbscanClustering(cluster_data_arr)
    cluster_info_dict[Y[2]] = [Y[1]]
    cluster_sil_dict[Y[2]] = Y[0]
    return cluster_sil_dict, cluster_info_dict

def evaluateBestCluster(cluster_sil_dict, cluster_info_dict):
    """
    Evaluates the best clustering method which achieved the highest silhouette score
    Sorts the LLA by which cluster they are in

    Parameters
    ----------

    cluster_sil_dict : Dict
        A dictionary where for each entry the key is the clustering method used, and the values is the silhouette score

    cluster_info_dict : Dict
        A dictionary where for each entry the key is the clustering method used, and the values is the labels of the clustering

    Returns
    -------
    group_dict : Dict
        A dictionary where for each entry the key is the group number and the value is a list with all the LLA's which belong to the group
        
    """
    # Finds the best clustering algorithm and creates a new dictionary with its stats
    group_dict = {}
    cluster_stats_dict = {}
    best_agl = max(cluster_sil_dict, key=cluster_sil_dict.get)
    best_labels = cluster_info_dict[best_agl]
    num_clustered = len(set(best_labels[0]))
    cluster_stats_dict = {"Algorithm Used " : [best_agl],
                          "Silhouette Score": [cluster_sil_dict[best_agl]],
                          "Number of Clusters": [num_clustered]}
    cluster_stats_df = pd.DataFrame(cluster_stats_dict)
    # Imports a dataframe of the LLA's 
    # Creates a dictionary which contains the different cluster groups as lists
    LLA_df = pd.read_excel("static\LLA's.xlsx")
    LLA_list = LLA_df["Lower tier local authorities"].tolist()
    #Even though the label are one list of items, as they are a numpy array they have to be accessed at index 0
    for index, item in enumerate(best_labels[0]):
        if item in group_dict:
            group_list = group_dict[item]
            group_list.append(LLA_list[index])
            group_dict[item] = group_list
        else:
            group_dict[item] = [LLA_list[index]]
    
            
    return group_dict ,cluster_stats_df

def createClusterLists(group_dict : dict):
    """
    Assigns a colour to each cluster depending on the value of that clusters key in the group_dict

    Parameters
    ----------
    group_dict : Dictionary
        A dictionary where for each entry the key is the group number and the value is a list with all the LLA's which belong to the group

    Returns
    -------
    colour_group_dict : Dictionary
        A dictionary where for each entry the key is the colour of the cluster and the value is the list with all the LLA's which belong to that cluster
    """
    # This creates a list of each cluster by the colour they use
    # This is not used to colour in the geopandas LLA map
    # It is used to display which LLA's are in what cluster by their colour
    colour_group_dict = {}
    for key in group_dict.keys():
        colour_key = colourSwitch(key)
        colour_group_dict[colour_key] = group_dict[key]
    return colour_group_dict


def colourSwitch(key):
    """
    A switch statement which determines the colour of a cluster on the map of the uk by its group number

    Parameters
    ----------
    key : int
        The number of the group
    
    Returns
    -------
    colour.get(key) : str
        The colour that is associated with that group number
    """
    # A switch statement to assign colours to clusters
    # -1 is for outlier values made by DBSCAN, it is always Black
    colour = {
        -1:"Black",
        0:"Coral",
        1:"White",
        2:"Green",
        3:"Pink",
        4:"Orange",
        5:"Yellow",
        6:"Lavender",
        7:"Grey",
        8:"Purple",
        9:"Indigo",
        10:"Darkgreen",
        11:"Brown",
        12:"Deeppink",
        13:"Navy",
        14:"Tan",
        15:"Turquoise",
        16:"Aquamarine",
        17:"Forestgreen",
        18:"Lightgrey",
        19:"Olive",
        20:"Magenta",
        21:"Darkred",
        22:"Lime",
        23:"khaki",
        24:"Beige",
        25:"Violet",
        26:"Steelblue",
        27:"Crimson",
        28:"Salmon",
        29:"Sienna",
        30:"Steelblue"
    }
    return colour.get(key)

def createColourDF(group_dict):
    """
    Creates a df where each row has an LLA and the colour it will be on the cluster map.
    The colour which is determine by which group it was clustered into

    Parameters
    ----------
    group_dict : Dict
        A dictionary where for each entry the key is the group number and the value is a list with all the LLA's which belong to the group

    Returns
    -------
        colour_df : DataFrame
            A dataframe where each row contains an LLA and the colour it will be on the cluster map
    """
    # Adds colour to the LLA map by adding a new attribute called colour
    # The colour is determined by the function colourSwitch()
    colour_dict = {}
    for key, items in group_dict.items():
        for lla in items:
            colour = colourSwitch(key)
            colour_dict[lla] = colour
    colour_df = pd.DataFrame.from_dict(colour_dict, orient='index', columns=["colour"])
    colour_df.reset_index(inplace=True)
    return colour_df

def displayClusterMap(colour_df):
    """
    Imports the geopandas shape file, merges the shape file with the colour df so each LLA is coloured according to its cluster
    The shape file is plotted and saved as a png to be displayed on the web application

    Parameters
    ----------
    colour_df : DataFrame
        A dataframe where each row contains an LLA and the colour it will be on the cluster map

    Returns
    -------
    None
    """
    # This loads the shape file used to display a cluster map of the LLA's
    LLA_shp = gpd.read_file("static\Ew_ltla_2022\ew_ltla_2022.shp")
    LLA_shp = pd.merge(LLA_shp, colour_df, left_on="name", right_on="index", how="inner" )
    fig, ax = plt.subplots(figsize=(35,25))
    ax.axis('off')
    # Plots the map
    LLA_shp.plot(color=LLA_shp["colour"],ax=ax, edgecolor="black")
    # The saved png file is loaded by the HTML of the webpage
    plt.savefig("static/cluster_map.png", bbox_inches="tight")
    # This is a separate png which is zoomed in on london by limiting the axis of the plot
    # This is becuase the London boroughs are hard to distinguish due to their size
    ax.set_xlim(500000,580000)
    ax.set_ylim(140000,230000)
    plt.savefig("static/cluster_map_london.png", bbox_inches="tight")
    

def splitDataFrameViaClusters(group_dict : dict,cluster_df_no_norm):
    """
    Splits the dataframe by the clustering results

    Parameters
    ----------
        n/a

    Returns
    -------
    split_cluster_dict : Dict
        A dictionary where for each entry the key is the group number and the values are the df rows of the LLA's within that group
    """
    # Splits the LLA's by their clusters
    # This is used for basic statistics and T-testing
    split_cluster_dict = {}
    for key, lla_list in group_dict.items():
        cluster_rows = cluster_df_no_norm[cluster_df_no_norm['Lower tier local authorities'].isin(lla_list)]
        split_cluster_dict[key] = cluster_rows
    return split_cluster_dict

def basicStats(split_cluster_dict : dict, cluster_df_no_norm : pd.DataFrame):
    """
    Displays the basic stats of all LLA's and the LLA's of each cluster

    Parameters
    ----------
    split_cluster_dict : Dict
        A dictionary where for each entry the key is the group number and the values are the df rows of the LLA's within that group

    Returns
    -------
    overall_stats : DataFrame
        The basic stats of all LLA's
        
    stats_dict : Dict
        A dictionary where for each entry the key is the colour of that cluster group and the values is the basic stats for that cluster group

    """
    # All these statistics will be shown to the user
    # Uses the .describe() function to get basic stats
    # This is to get the stats of all the LLA's together
    # This will be used for T-testing as we need to compare the mean of each cluster to the mean overall to determine significance
    # We transpose the stats to make the count mean and std as attributes and the data categories as rows
    stats_dict = {}
    overall_stats = cluster_df_no_norm.describe()
    overall_stats = overall_stats.loc[["count","mean","std"]].T.reset_index()
    overall_stats = overall_stats.rename(columns={"index":"Data Attributes"})
    # This creates basic stats for each individual cluster
    # These will be used for T-testing as each clusters mean will need to be compared to the overall mean to determine significance
    for key, clustered_df in split_cluster_dict.items():
        cluster_colour = colourSwitch(key)
        stats = clustered_df.describe()
        stats = stats.loc[["count","mean","std"]].T.reset_index()
        stats_dict[cluster_colour] = stats
    return overall_stats, stats_dict

def preProcessTest(overall_stats : pd.DataFrame, stats_dict : dict):
    """
    Performs T testing on the means of each attribute from each cluster to determine
    if they are signifcantly significant from the means of each attribute from the dataset as a whole

    Parameters
    ----------

    overall_stats : DataFrame
        The basic stats of all LLA's
        
    stats_dict : Dict
        A dictionary where for each entry the key is the colour of that cluster group and the values is the basic stats for that cluster group

    Returns
    -------
        None
    """
    # Iterates over the individual statistics of each clusters and performed T-test on them compared to the mean of all LLA's
    for keys ,values in stats_dict.items():
        pvalue_list = []
        significance_dict = {}
        significance_list = []
        for index, rows in values.iterrows():
            tstat, pvalue = ttest_ind_from_stats(overall_stats.loc[index]["mean"],
                                          overall_stats.loc[index]["std"],
                                          overall_stats.loc[index]["count"],
                                          rows["mean"], rows["std"], rows["count"])
            pvalue_list.append(pvalue)
            if pvalue < 0.05:
                significance_list.append(True)
            elif pvalue >= 0.05:
                significance_list.append(False)
            else:
                # This is needed for when a significance value cannot be determined 
                # This occurs for certain data categories e.g. "0 people live in household" for most LLA's this has a value of 0 causing an error 
                # as no significance value can be determined
                significance_list.append("n/a")
        significance_dict["P-Value"] = pvalue_list
        significance_dict["significance T/F"] = significance_list
        significance_df = pd.DataFrame(significance_dict)
        # Matches the index of the stats df with the significance df so they concatenate correctly
        significance_df.index = stats_dict[keys].index
        # Concatenates the significant dataframe to the stat dataframe
        # This will be displayed by the user
        stats_dict[keys] = pd.concat([stats_dict[keys], significance_df],axis=1)
        stats_dict[keys] = stats_dict[keys].rename(columns={"index":"Data Attributes"})

    return stats_dict

def convertDataFrameToHTML(stats_dict):
    """
    Converts a panda's dataframe into a HTML table

    Parameters
    ----------
    stats_dict : Dictionary
        A dictionary which contains the count, mean, std, p-value and significance boolean for each individual cluster

    Returns
    -------
    stats_dict_html : Dictionary
        A dictionary which contains the HTML converted stats table of all clusters
    
    """
    # Converts the stats of each cluster into an HTML format so it can be displayed on the web application
    stats_dict_html = {}
    for key, value in stats_dict.items():
        html_table = value.to_html()
        stats_dict_html[key] = html_table
    return stats_dict_html
