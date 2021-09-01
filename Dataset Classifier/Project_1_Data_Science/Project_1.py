# @ author
#                   Name: Maor Mohav , ID: 316142363

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

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

##
#################  Global Variabels  #################
##
dic = None
train_file = None
test_file = None


##
#################  Delete rows with missing classification value.  #################
##
def Delete_Rows(data):
    data['class'].replace('', np.nan, inplace=True)
    data.dropna(subset=['class'], inplace=True)


##
#################  Returns true if the col values are numeric , else returns false.  #################
##
def numeric(j):
    for i in j:
        if type(i) == str and i != '':
            return False

    return True

##
#################  Creating array of means to all the columns depending of 'yes' or 'no'.  #################
##
def D_Array(data):
    a = []
    no = data[data['class'] == 'no'].dropna(axis=0)
    yes = data[data['class'] == 'yes'].dropna(axis=0)

    for i in data.columns:
        if numeric(data[i]):
            a.append([yes[i].mean(), no[i].mean()])
        else:
            a.append([yes[i].value_counts().index[0], no[i].value_counts().index[0]])

    return a

##
#################  Complete the missing values of every columns , depending on user's choose.  #################
##
def Missing_Data(data):
    a = D_Array(train_file)
    c = data.columns.tolist()

    if dic['missing_data'] == 'By all data':
        for i in data.columns:
            if numeric(data[i]) == True:                                                        # fills by all data method.
                data[i].fillna(train_file[i].mean(), inplace=True)
            else:
                data[i].fillna(train_file[i].value_counts().index[0], inplace=True)

    else:
        for i, rows in data.iterrows():                                                         # fills by classification data.
            for b in data.columns:
                if pd.isnull(data[b][i]) and data['class'][i] == 'yes':
                    data.at[i, str(b)] = a[c.index(b)][0]
                elif pd.isnull(data[b][i]) and data['class'][i] == 'no':
                    data.at[i, str(b)] = a[c.index(b)][1]


##
#################  Convert all columns to float numbers if the column is numeric.  #################
##
def convert_data():
    for i in train_file.columns:
        if numeric(train_file[i]):
            train_file[i] = pd.to_numeric(train_file[i], downcast='float')
            test_file[i] = pd.to_numeric(test_file[i], downcast='float')



##
#################  Creating Normalization to columns data - numeric columns.  #################
##
def Normalization(train_file, test_file):
    if dic['normalization'] != 'No':
        for i in train_file.columns:
            if numeric(train_file[i]):
                avg = train_file[i].mean()
                std = train_file[i].std()

                for j in range(0, len(train_file[i])):
                    train_file.at[j, str(i)] = ((train_file[i][j] - avg) / std)

                for j in range(0, len(test_file[i])):
                    test_file.at[j, str(i)] = ((test_file[i][j] - avg) / std)

##
#################  Discritization - built in .  #################
##
def Equal_width_binning(data, bins, label):
    for i in data:
        if numeric(data[i]):
            data[i] = pd.cut(data[i], bins, labels=label)

def Equal_frequency_binning(data, bins):
        for i in data:
            if numeric(data[i]):
                data[i] = pd.qcut(data[i], bins, duplicates='drop')
                temp_categories = {value: j for j, value in enumerate(data[i].sort_values().unique(), start=1)}
                data[i] = data[i].cat.rename_categories(temp_categories)


# Equal-frequency discretization
##
#################  Discritization - our implementation.  #################
##
def Equal_frequency_binning_implementation():
    def Equal_frequency(data, test, columns, m):

        a = []

        for i in data[columns]:
            a.append(i)

        a.sort()

        length = len(a)
        n = int(length / m)
        arri = []
        for i in range(0, m):
            arr = []
            for j in range(i * n, (i + 1) * n):
                if j >= length:
                    break
                arr = arr + [a[j]]
            arri.append(arr)

        for i in range(0, len(arri)):
            arri[i] = list(set(arri[i]))

        for i in range(0, len(test[columns])):
            for j in range(0, len(arri)):
                if test[columns][i] in arri[j]:
                    test.at[i, str(columns)] = j

        for i in range(0, len(data[columns])):
            for j in range(0, len(arri)):
                if data[columns][i] in arri[j]:
                    data.at[i, str(columns)] = j

    for i in train_file.columns:
        if numeric(train_file[i]):
            Equal_frequency(train_file, test_file, i, dic['number_of_bins'])


# Equal-width discretization
##
#################  Discritization - our implementation.  #################
##
def Equal_width_binning_implementation():
    def Equal_width(data, test, columns, m):

        a = []

        for i in data[columns]:
            a.append(i)

        a.sort()
        w = int((max(a) - min(a)) / m) + 1
        min1 = min(a)
        arr = []
        for i in range(0, m + 1):
            arr = arr + [min1 + w * i]
        arri = []

        for i in range(0, m):
            temp = []

            for j in a:
                if j >= arr[i] and j <= arr[i + 1]:
                    temp += [j]

            arri += [temp]

        for i in range(0, len(arri)):
            arri[i] = list(set(arri[i]))

        for i in range(0, len(test[columns])):
            for j in range(0, len(arri)):
                if test[columns][i] in arri[j]:
                    test.at[i, str(columns)] = j

        for i in range(0, len(data[columns])):
            for j in range(0, len(arri)):
                if data[columns][i] in arri[j]:
                    data.at[i, str(columns)] = j

    for i in train_file.columns:
        if numeric(train_file[i]):
            Equal_width(train_file, test_file, i, dic['number_of_bins'])


##
#################  Discritization the data by user's input.  #################
##
def Discritization(data):


    if dic['discritization'] != 'No discritization':
        bins = dic['number_of_bins']

        label = []

        for i in range(0, bins):
            label.append(i)

    if dic['discritization'] == 'Equal Width Binning':
        Equal_width_binning(data, bins, label)

    elif dic['discritization'] == 'Equal Frequency Binning':
        Equal_frequency_binning(data, bins)

    elif dic['discritization'] == 'Equal Width Binning-Implementation':
        Equal_width_binning_implementation()

    elif dic['discritization'] == 'Equal Frequency Binning-Implementation':
        Equal_frequency_binning_implementation()

    elif dic['discritization'] == 'Discritization based Entropy':

        for i in data.columns:
            if numeric(data[i]):
                val = np.array(data[i])
                new_val = val.astype(int)
                IG = drv.entropy(new_val)
                for j in range(0, len(data[i])):
                    if data[i][j] < IG:
                        data.at[j, str(i)] = 0
                    else:
                        data.at[j, str(i)] = 1


##
#################  Encode the columns that are not numeric.  #################
##
def Encoder(data, dat):
    for i in data.columns:
        if not numeric(data[i]):
            le = LabelEncoder()
            data[i] = le.fit_transform(data[i])
            dat[i] = le.fit_transform(dat[i])


##
#################  Naive bayes classifier - built in .  #################
##
def Naive_bayes():
    X_train = train_file.iloc[:, : -1]
    Y_train = train_file.iloc[:, -1]
    X_test = test_file.iloc[:, : -1]
    Y_test = test_file.iloc[:, -1]

    print("------------------------------------------------------------------------------")
    print("------------------------------Naive-bayes- Model------------------------------")
    print()
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != y_pred).sum()))
    accuracy = accuracy_score(Y_test, y_pred) * 100
    print("precentence: ", accuracy)
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, y_pred))
    print()
    print("Report: ")
    print()
    print(classification_report(Y_test, y_pred))

    pickle.dump(gnb, open('Naive_bayes.sav', 'wb'))                     ########################## pickle #############################

##
#################  Decision tree classifier - built in .  #################
##
def Decision_Tree():
    X_train = train_file.iloc[:, : -1]
    Y_train = train_file.iloc[:, -1]
    X_test = test_file.iloc[:, : -1]
    Y_test = test_file.iloc[:, -1]

    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    pred = tree.predict(X_test)

    print("------------------------------------------------------------------------------")
    print("------------------------------------ID3- Model--------------------------------")
    print()
    print("The prediction accuracy is: ", tree.score(X_test, Y_test) * 100, "%")
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, pred))
    print()
    print("Report: ")
    print()
    print(classification_report(Y_test, pred))

    pickle.dump(tree, open('Id3.sav', 'wb'))                                                          ########################## pickle #############################


##
#################  Naive bayes classifier - our implementation .  #################
##
def Naive_bayes_implementation():

    py = 0
    pn = 0

    total = len(test_file['class'])
    print("in processing..................")
    correct = 0
    y = 0
    n = 0

    for i in train_file['class']:
        if i == 1:
            y = y + 1
        else:
            n = n + 1

    py = y / len(train_file['class'])
    pn = n / len(train_file['class'])

    for i, rows in test_file.iterrows():

        arryes = []
        arrno = []

        for j in test_file.columns:
            dyes = train_file[train_file[j] == test_file[j][i]]
            dyes = train_file[train_file['class'] == 1]

            arryes.append(len(dyes['class']) / y)

            dno = train_file[train_file[j] == test_file[j][i]]
            dno = train_file[train_file[j] == 0]

            arrno.append(len(dno['class']) / n)

        yes = 0
        no = 0

        for j in arryes:
            if yes == 0:
                yes = j
            else:
                yes = yes * j

        yes = yes * py

        for j in arrno:
            if no == 0:
                no = j
            else:
                no = no * j

        no = no * pn

        if yes > no and 1 == test_file['class'][i]:
            correct = correct + 1

        elif no > yes and 0 == test_file['class'][i]:
            correct = correct + 1

    print("------------------------------------------------------------------------------")
    print("------------------------------Naive-bayes- Model------------------------------")
    print()
    print("Accuracy of the model: ", (correct / len(test_file['class'])) * 100, "%")


##
#################  Decision tree classifier - our implementation .  #################
##
def Decision_Tree_implementation():
    dataset = train_file
    test_dataset = test_file

    def entropy(data_set):
        """ this function calculate data set entropy """
        Probability_set = []

        for i in data_set:  # create a list of Probability
            counter = 0
            for j in data_set:
                if j == i:
                    counter += 1
            Probability_set.append(counter / len(data_set))

        return sum(list(map(lambda x: (-1) * (x * log(x, 2)), Probability_set)))  # return entropy

    def InfoGain(data, split_attribute_name, target_name="class"):

        # Entropy of the total dataset
        total_entropy = entropy(data[target_name])

        ##Calculate the entropy of the dataset
        values, counts = np.unique(data[split_attribute_name], return_counts=True)

        # weighted entropy
        W_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(
            data.where(data[split_attribute_name] == values[i]).dropna()[target_name]) for i in range(len(values))])

        # Calculate the information gain
        Info_Gain = total_entropy - W_Entropy
        return Info_Gain

    def ID3(data, originaldata, features, target_attribute="class", Parent_Node=None):

        if len(np.unique(data[target_attribute])) <= 1:
            return np.unique(data[target_attribute])[0]


        elif len(data) == 0:
            return np.unique(originaldata[target_attribute])[
                np.argmax(np.unique(originaldata[target_attribute], return_counts=True)[1])]


        elif len(features) == 0:
            return Parent_Node

        else:

            Parent_Node = np.unique(data[target_attribute])[
                np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]

            item = [InfoGain(data, feature, target_attribute) for feature in features]
            index_OfBestFeature = np.argmax(item)
            Bestfeature = features[index_OfBestFeature]

            # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
            # gain in the first run
            tree = {Bestfeature: {}}

            features = [i for i in features if i != Bestfeature]

            for value in np.unique(data[Bestfeature]):
                value = value

                sub_data = data.where(data[Bestfeature] == value).dropna()

                # Call the ID3 algorithm for each of those sub_datasets with the new parameters, Here the recursion comes in!
                subtree = ID3(sub_data, dataset, features, target_attribute, Parent_Node)

                tree[Bestfeature][value] = subtree

            return (tree)

    def predict(query, tree, default=1):

        for key in list(query.keys()):
            if key in list(tree.keys()):

                try:
                    result = tree[key][query[key]]
                except:
                    return default

                result = tree[key][query[key]]

                if isinstance(result, dict):
                    return predict(query, result)

                else:
                    return result

    def testing(data, tree):

        queries = data.iloc[:, :-1].to_dict(orient="records")

        predicted = pd.DataFrame(columns=["predicted"])

        for i in range(len(data)):
            predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)

        print("------------------------------------------------------------------------------")
        print("----------------------------------ID3- Model----------------------------------")
        print()

        print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["class"]) / len(data)) * 100, '%')

    train_dataset = dataset.iloc[:100].reset_index(drop=True)

    """
    Train the tree, Print the tree and predict the accuracy
    """
    tree = ID3(train_dataset, train_dataset, train_dataset.columns[:-1])
    testing(test_dataset, tree)



##
#################  Classification model - by user's inputs  .  #################
##
def Model(train_file, test_file):

    if dic['model_type'] == 'Naive bayes':
        Naive_bayes()

    elif dic['model_type'] == 'Naive bayes-Implementation':
        Naive_bayes_implementation()

    elif dic['model_type'] == 'Decision tree':
        Decision_Tree()

    elif dic['model_type'] == 'Decision tree-Implementation':
        Decision_Tree_implementation()


##
#################  Knn classifier - built in .  #################
##
def Knn():
    X_train = train_file.iloc[:, : -1]
    Y_train = train_file.iloc[:, -1]
    X_test = test_file.iloc[:, : -1]
    Y_test = test_file.iloc[:, -1]

    arr = []

    for i in range(2, 10):
        classifier = KNeighborsClassifier(n_neighbors=i, p=2, metric='euclidean')
        classifier.fit(X_train, Y_train)
        y_pred = classifier.predict(X_test)
        arr.append(accuracy_score(Y_test, y_pred))

    index = 0
    mx = 0

    for i in range(len(arr)):
        if arr[i] >= mx:
            mx = arr[i]
            index = i

    best_n_neighbors = (index + 2)

    classifier = KNeighborsClassifier(n_neighbors=best_n_neighbors - 1, p=2, metric='euclidean')
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    print("------------------------------------------------------------------------------")
    print("--------------------------------KNN - Classifier------------------------------")
    print()
    print("The best n_neighbors is: ", best_n_neighbors)
    print("The prediction accuracy is: ", mx * 100)
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, y_pred))
    print()
    print("Report: ")
    print()
    print(classification_report(Y_test, y_pred))

    pickle.dump(classifier, open('Knn.sav', 'wb'))

##
#################  K means classifier - built in .  #################
##
def K_means():
    X_train = train_file.iloc[:, : -1]
    Y_train = train_file.iloc[:, -1]
    X_test = test_file.iloc[:, : -1]
    Y_test = test_file.iloc[:, -1]
    arr = []

    for i in range(2, 10):
        classifier = KMeans(n_clusters=i)
        classifier.fit(X_train, Y_train)
        y_pred = classifier.predict(X_test)
        arr.append(accuracy_score(Y_test, y_pred))

    index = 0
    mx = 0

    for i in range(len(arr)):
        if arr[i] >= mx:
            mx = arr[i]
            index = i

    best_n_clusters = (index + 2)

    classifier = KMeans(n_clusters=best_n_clusters)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)


    print("------------------------------------------------------------------------------")
    print("--------------------------------KMeans - Classifier---------------------------")
    print()
    print("The best n_clusters is: ", best_n_clusters)
    print("The prediction accuracy is: ", mx * 100)
    print("Confusion matrix: ")
    print(confusion_matrix(Y_test, y_pred))
    print()
    print("Report: ")
    print()
    print(classification_report(Y_test, y_pred))

    pickle.dump(classifier, open('K_means.sav', 'wb'))

##
#################  Initialize the classifier system on the given dataset.  #################
##
def initialize():
    Delete_Rows(train_file)                  ###### delete rows
    Delete_Rows(test_file)

    Missing_Data(train_file)                 ###### replace missing data
    Missing_Data(test_file)
    convert_data()
    Normalization(train_file, test_file)    ###### rnormalization

    Discritization(train_file)              ###### discritization
    Discritization(test_file)

    Encoder(train_file, test_file)          ###### encoder

    train_file.to_csv('train_file_clean.csv')          ########################## saving files to csv   #############################
    test_file.to_csv('test_file_clean.csv')

    Model(train_file, test_file)                       ###### model.

    Knn()                                              ###### knn and k-means.
    K_means()
##
#################  GUI SYSTEM  #################
##
def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    unknown_support.set_Tk_var()
    top = Toplevel1 (root)
    unknown_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    #rt = root
    root = rt
    w = tk.Toplevel (root)
    unknown_support.set_Tk_var()
    top = Toplevel1 (w)
    unknown_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1098x655+376+207")
        top.minsize(120, 1)
        top.maxsize(3844, 1061)
        top.resizable(1,  1)
        top.title("Classification Model")
        top.configure(background="#ffffdf")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.menubar = tk.Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)

        self.title = tk.Label(top)
        self.title.place(relx=0.182, rely=0.046, height=51, width=694)
        self.title.configure(activebackground="#f9f9f9")
        self.title.configure(activeforeground="black")
        self.title.configure(background="#c6c6ff")
        self.title.configure(disabledforeground="#a3a3a3")
        self.title.configure(font="-family {Microsoft YaHei UI Light} -size 20 -weight bold")
        self.title.configure(foreground="#000000")
        self.title.configure(highlightbackground="#d9d9d9")
        self.title.configure(highlightcolor="black")
        self.title.configure(text='''Welcome To Classification Model Creation''')

        self.train_path_label = tk.Label(top)
        self.train_path_label.place(relx=0.018, rely=0.183, height=21, width=163)

        self.train_path_label.configure(activebackground="#f9f9f9")
        self.train_path_label.configure(activeforeground="black")
        self.train_path_label.configure(background="#d9d9d9")
        self.train_path_label.configure(disabledforeground="#a3a3a3")
        self.train_path_label.configure(font="-family {Microsoft YaHei UI Light} -size 12 -weight bold")
        self.train_path_label.configure(foreground="#000000")
        self.train_path_label.configure(highlightbackground="#d9d9d9")
        self.train_path_label.configure(highlightcolor="black")
        self.train_path_label.configure(text='''Path's training file:''')

        self.train_path = tk.Text(top)
        self.train_path.place(relx=0.182, rely=0.183, relheight=0.052
                , relwidth=0.696)
        self.train_path.configure(background="white")
        self.train_path.configure(font="TkTextFont")
        self.train_path.configure(foreground="black")
        self.train_path.configure(highlightbackground="#d9d9d9")
        self.train_path.configure(highlightcolor="black")
        self.train_path.configure(insertbackground="black")
        self.train_path.configure(selectbackground="blue")
        self.train_path.configure(selectforeground="white")
        self.train_path.configure(wrap="word")

        self.test_path_label = tk.Label(top)
        self.test_path_label.place(relx=0.018, rely=0.305, height=21, width=162)
        self.test_path_label.configure(activebackground="#f9f9f9")
        self.test_path_label.configure(activeforeground="black")
        self.test_path_label.configure(background="#d9d9d9")
        self.test_path_label.configure(disabledforeground="#a3a3a3")
        self.test_path_label.configure(font="-family {Microsoft YaHei UI Light} -size 12 -weight bold")
        self.test_path_label.configure(foreground="#000000")
        self.test_path_label.configure(highlightbackground="#d9d9d9")
        self.test_path_label.configure(highlightcolor="black")
        self.test_path_label.configure(text='''Path's test file:''')

        self.Test_path = tk.Text(top)
        self.Test_path.place(relx=0.182, rely=0.305, relheight=0.052
                , relwidth=0.699)
        self.Test_path.configure(background="white")
        self.Test_path.configure(font="TkTextFont")
        self.Test_path.configure(foreground="black")
        self.Test_path.configure(highlightbackground="#d9d9d9")
        self.Test_path.configure(highlightcolor="black")
        self.Test_path.configure(insertbackground="black")
        self.Test_path.configure(selectbackground="blue")
        self.Test_path.configure(selectforeground="white")
        self.Test_path.configure(wrap="word")

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=0.437, rely=0.885, height=24, width=157)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")


        self.Button1.configure(command=self.Submit)


        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(font="-family {Microsoft YaHei Light} -size 9 -weight bold")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Button''')

        self.missing_data_label = tk.Label(top)
        self.missing_data_label.place(relx=0.018, rely=0.412, height=21
                , width=252)
        self.missing_data_label.configure(activebackground="#f9f9f9")
        self.missing_data_label.configure(activeforeground="black")
        self.missing_data_label.configure(background="#d9d9d9")
        self.missing_data_label.configure(cursor="fleur")
        self.missing_data_label.configure(disabledforeground="#a3a3a3")
        self.missing_data_label.configure(font="-family {Microsoft YaHei UI Light} -size 12 -weight bold")
        self.missing_data_label.configure(foreground="#000000")
        self.missing_data_label.configure(highlightbackground="#d9d9d9")
        self.missing_data_label.configure(highlightcolor="black")
        self.missing_data_label.configure(text='''Missing Data Completion:''')

        self.normalization_label = tk.Label(top)
        self.normalization_label.place(relx=0.018, rely=0.504, height=21
                , width=252)
        self.normalization_label.configure(activebackground="#f9f9f9")
        self.normalization_label.configure(activeforeground="black")
        self.normalization_label.configure(background="#d9d9d9")
        self.normalization_label.configure(disabledforeground="#a3a3a3")
        self.normalization_label.configure(font="-family {Microsoft YaHei UI Light} -size 12 -weight bold")
        self.normalization_label.configure(foreground="#000000")
        self.normalization_label.configure(highlightbackground="#d9d9d9")
        self.normalization_label.configure(highlightcolor="black")
        self.normalization_label.configure(text='''Normalization:''')

        self.discritization_label = tk.Label(top)
        self.discritization_label.place(relx=0.018, rely=0.595, height=21
                , width=252)
        self.discritization_label.configure(activebackground="#f9f9f9")
        self.discritization_label.configure(activeforeground="black")
        self.discritization_label.configure(background="#d9d9d9")
        self.discritization_label.configure(disabledforeground="#a3a3a3")
        self.discritization_label.configure(font="-family {Microsoft YaHei UI Light} -size 12 -weight bold")
        self.discritization_label.configure(foreground="#000000")
        self.discritization_label.configure(highlightbackground="#d9d9d9")
        self.discritization_label.configure(highlightcolor="black")
        self.discritization_label.configure(text='''Discritization:''')

        self.bins_label = tk.Label(top)
        self.bins_label.place(relx=0.565, rely=0.595, height=21, width=143)
        self.bins_label.configure(activebackground="#f9f9f9")
        self.bins_label.configure(activeforeground="black")
        self.bins_label.configure(background="#d9d9d9")
        self.bins_label.configure(cursor="fleur")
        self.bins_label.configure(disabledforeground="#a3a3a3")
        self.bins_label.configure(font="-family {Microsoft YaHei UI Light} -size 9 -weight bold")
        self.bins_label.configure(foreground="#000000")
        self.bins_label.configure(highlightbackground="#d9d9d9")
        self.bins_label.configure(highlightcolor="black")
        self.bins_label.configure(text='''Number of bins:''')

        self.bins = tk.Spinbox(top, from_=1.0, to=100.0)
        self.bins.place(relx=0.71, rely=0.595, relheight=0.029, relwidth=0.123)
        self.bins.configure(activebackground="#f9f9f9")
        self.bins.configure(background="white")
        self.bins.configure(buttonbackground="#d9d9d9")
        self.bins.configure(disabledforeground="#a3a3a3")
        self.bins.configure(font="TkDefaultFont")
        self.bins.configure(foreground="black")
        self.bins.configure(highlightbackground="black")
        self.bins.configure(highlightcolor="black")
        self.bins.configure(insertbackground="black")
        self.bins.configure(selectbackground="blue")
        self.bins.configure(selectforeground="white")
        self.bins.configure(textvariable=unknown_support.spinbox)

        self.model_label = tk.Label(top)
        self.model_label.place(relx=0.018, rely=0.687, height=21, width=252)
        self.model_label.configure(activebackground="#f9f9f9")
        self.model_label.configure(activeforeground="black")
        self.model_label.configure(background="#d9d9d9")
        self.model_label.configure(cursor="fleur")
        self.model_label.configure(disabledforeground="#a3a3a3")
        self.model_label.configure(font="-family {Microsoft YaHei UI Light} -size 12 -weight bold")
        self.model_label.configure(foreground="#000000")
        self.model_label.configure(highlightbackground="#d9d9d9")
        self.model_label.configure(highlightcolor="black")
        self.model_label.configure(text='''Model type:''')

        self.model_type = ttk.Combobox(top)
        self.model_type.place(relx=0.282, rely=0.687, relheight=0.032
                , relwidth=0.23)
        self.value_list = ['Naive bayes','Naive bayes-Implementation','Decision tree','Decision tree-Implementation']
        self.model_type.configure(values=self.value_list)
        self.model_type.configure(takefocus="")

        self.normalization = ttk.Combobox(top)
        self.normalization.place(relx=0.282, rely=0.504, relheight=0.032
                , relwidth=0.23)
        self.value_list2 = ['Yes','No',]
        self.normalization.configure(values=self.value_list2)
        self.normalization.configure(takefocus="")
        self.normalization.configure(cursor="fleur")

        self.TCombobox2 = ttk.Combobox(top)
        self.TCombobox2.place(relx=0.282, rely=0.412, relheight=0.032
                , relwidth=0.228)
        self.value_list3 = ['By classification data.','By all data',]
        self.TCombobox2.configure(values=self.value_list3)
        #self.TCombobox2.configure(textvariable=unknown_support.combobox)
        self.TCombobox2.configure(takefocus="")

        self.TCombobox1 = ttk.Combobox(top)

        self.TCombobox1.place(relx=0.282, rely=0.595, relheight=0.032
                , relwidth=0.228)
        self.value_list4 = ['No discritization','Equal Width Binning','Equal Width Binning-Implementation','Equal Frequency Binning','Equal Frequency Binning-Implementation','Discritization based Entropy',]
        self.TCombobox1.configure(values=self.value_list4)
        #self.TCombobox1.configure(textvariable=unknown_support.combobox)
        self.TCombobox1.configure(takefocus="")

        #self.TCombobox1.configure(command= self.hide_or_show)
        #self.TCombobox1.activated.connect(self.hide_or_show)
        self.TCombobox1.current(0)
        self.TCombobox1.bind("<<ComboboxSelected>>", self.hide_or_show)

        self.bins.place_forget()
        self.bins_label.place_forget()

    def hide_or_show(self, eventObject):

        if self.TCombobox1.get() == 'No discritization':
            self.bins.place_forget()
            self.bins_label.place_forget()
        else:
            self.bins.pack()
            self.bins.place(relx=0.71, rely=0.595, relheight=0.029, relwidth=0.123)
            self.bins_label.pack()
            self.bins_label.place(relx=0.565, rely=0.595, height=21, width=143)


    ##
    #################  Creating a dictionary that includes all the user's input.  #################
    ##
    def Submit(self):

        global dic, train_file , test_file




        if self.TCombobox1.get() == 'No discritization':
            self.dic = {'train_path': self.train_path.get("1.0", "end-1c"),
                        'test_path': self.Test_path.get("1.0", "end-1c"), 'missing_data': self.TCombobox2.get(),
                        'normalization': self.normalization.get(), 'discritization': self.TCombobox1.get(),
                        'number_of_bins': None, 'model_type': self.model_type.get()}
        else:
            self.dic = {'train_path': self.train_path.get("1.0", "end-1c"), 'test_path': self.Test_path.get("1.0", "end-1c"), 'missing_data': self.TCombobox2.get(),
                        'normalization': self.normalization.get(), 'discritization': self.TCombobox1.get(), 'number_of_bins': int(self.bins.get()), 'model_type': self.model_type.get()}

        dic = self.dic
        train_file = pd.read_csv(dic['train_path'])
        test_file = pd.read_csv(dic['test_path'])

        initialize()

#
############################ main ###########################
#
if __name__ == '__main__':
    vp_start_gui()





