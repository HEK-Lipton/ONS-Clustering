# ONS-Clustering
A python web application which uses clustering to group together similar counties on various socio-economic factors. These results are then visually displayed on a cluster map to the user. The user is able to select which combination of data catergories to cluster on.

-------------------------------------------

In this respostory there are two files :

1. Offline Data Pre-processing
2. Flask

Offline Data Pre-processing contains the python script which pre-processed each individual 2021 ONS census data catergory and converted it into a .csv file that is stored on the website

Flask contains the python web application, which takes the pre-processed data and runs multiple clustering algorithms on it. After finding the best results the cluster results, map and relevant statistics are shown to the user.

To run the web application open the app.py file and run it. A local host will appear in your terminal click it to open the website.
