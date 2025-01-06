from flask import Blueprint, render_template, request, jsonify, redirect
from flask import url_for
from functions import dataframeCreation
from functions import main_df,main_df_no_norm,df_index_dict, mainClusterLoop, preCluster, evaluateBestCluster, createColourDF, displayClusterMap, splitDataFrameViaClusters, dataframeCreationNoNorm, basicStats, preProcessTest, convertDataFrameToHTML, createClusterLists


views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html")

@views.route("/about")
def about():
    return render_template("about.html")

@views.route("/methods")
def methods():
    return render_template("methods.html")

@views.route("/submit", methods = ["POST"])
def submit():
    # This gets the list of data categories the user requested
    results = request.form.getlist('categories')

    # Creates a custom data frame with the categories the user selected
    df = dataframeCreation(results)

    # This Clusters the user requested data over four clustering methods
    # Each method is hyperparameter trained and the best results from each are stored
    cluster_sil_dict, cluster_info_dict = mainClusterLoop(df)

    # The clustering method with the highest silhouette score is chosen
    group_dict, cluster_stats_df = evaluateBestCluster(cluster_sil_dict, cluster_info_dict)

    # A dataframe and a Dictionary are created which contain the colour of each LLA depending on the
    # cluster it belings to
    colour_df = createColourDF(group_dict)
    colour_group_dict = createClusterLists(group_dict)

    # This displays and exports a png of the two images of the LLA cluster map
    displayClusterMap(colour_df)

    # A normalised copy of the user requested data is created as it used to display the final statistics for 
    # each cluster
    df_no_norm = dataframeCreationNoNorm(results)

    # The LLA's are split by there clusters in order to display their individual stats
    # and to carry out T-testing
    split_cluster_dict = splitDataFrameViaClusters(group_dict,df_no_norm)

    # The basic stats of all clusters are computed
    overall_stats, stats_dict = basicStats(split_cluster_dict,df_no_norm)

    # T-test are conducted comparing the individual cluster results to the results of all LLA's
    # To determine if the clusters are significant
    stats_dict = preProcessTest(overall_stats,stats_dict)
    # The statistics dictionary is converted to HTML to be displayed on the web application
    stats_dict_html = convertDataFrameToHTML(stats_dict)

    # The results of the clustering are sent to the html page and the website is rendered
    return render_template("py.html", stats=overall_stats.to_html(), data=stats_dict_html, colour_dict=colour_group_dict, cluster_info=cluster_stats_df.to_html())
    

