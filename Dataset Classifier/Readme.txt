  ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______                                          
 |______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|                                         
  _____           _           _      __              _____        _             _____      _                                                    
 |  __ \         (_)         | |    /_ |            |  __ \      | |           / ____|    (_)                                                              
 | |__) | __ ___  _  ___  ___| |_    | |   ______   | |  | | __ _| |_ __ _    | (___   ___ _  ___ _ __   ___ ___                                           
 |  ___/ '__/ _ \| |/ _ \/ __| __|   | |  |______|  | |  | |/ _` | __/ _` |    \___ \ / __| |/ _ \ '_ \ / __/ _ \                                          
 | |   | | | (_) | |  __/ (__| |_    | |            | |__| | (_| | || (_| |    ____) | (__| |  __/ | | | (_|  __/                                          
 |_|   |_|  \___/| |\___|\___|\__|   |_|            |_____/ \__,_|\__\__,_|   |_____/ \___|_|\___|_| |_|\___\___|                                          
                _/ |                                                                                                                                       
  ______ ______|__/___ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______                                          
 |______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|  

 Authors:
 --------
                   Name: Maor Mohav , ID: 316142363
                   Name: Ariel Epshtein , ID: 316509504
                   Name: Omer Ben David , ID: 316344449

 Programming Enviorment:
 -----------------------
		   
		   Pycharm 2020.2.5 


 Installation packages and imports:
 ----------------------------------   

		   1. Import folder of the project in PYCHARM : Project_1_Data_Science
                   2. python -m pip install --upgrade pip
	           3. pip install pandas / python -m pip install pandas
	           4. pip install scikit-learn
		   5. pip install pyitlib   

 Project files:
 --------------
		   1.Project_1.py
                   2.unknown_support.py   (This file needs to be in the same folder with Project_1.py file , this file support the GUI system)


 Libraries:
 ----------
		   import sys
		   import Tkinter as tk
		   import ttk
		   import unknown_support
		   import pandas as pd
		   import numpy as np
		   from sklearn.preprocessing import LabelEncoder
		   from pyitlib import discrete_random_variable as drv
	           from math import log
		   from sklearn.tree import DecisionTreeClassifier
		   from sklearn.naive_bayes import GaussianNB
		   from sklearn.metrics import accuracy_score
		   from sklearn.neighbors import KNeighborsClassifier
                   from sklearn.metrics import confusion_matrix, classification_report
		   from sklearn.cluster import KMeans
		   import pickle


 Addition notes:
 ---------------  

                   1.After running the model , the files will be saved at the same folder that the project file exist.
	           2.Running the model with built-in libraries will take 1-2 minutes.
		     Running the model with our implementation will take 7-15 minutes.
                   
		   
   