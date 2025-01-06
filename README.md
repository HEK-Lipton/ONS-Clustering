# ONS-Clustering
A python web application which uses clustering to group together similar counties on various socio-economic factors. These results are then visually displayed on a cluster map to the user. The user is able to select which combination of data catergories to cluster on.

-------------------------------------------

In this respostory there are two files :

1. Data Pre-processing
2. Flask

Data Pre-processing contains the python script which pre-processed each individual 2021 ONS census data catergory and converted it into a .csv file that is stored on the website. To run it open the file and run the code.ipynb file. This will produce two files main_df.csv and main_df_no_norm.csv.

Flask contains the python web application, which takes the pre-processed data and runs multiple clustering algorithms on it. After finding the best results the cluster results, map and relevant statistics are shown to the user.

To run the web application open the flask folder in your IDE and run the app.py file. A local host will appear in your terminal click it to open the website.
