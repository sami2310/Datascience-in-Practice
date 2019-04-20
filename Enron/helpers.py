import os
import itertools

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from email.parser import Parser

import string
from nltk.corpus import stopwords


####### ####### Data existence ####### #######

def check_existance(file_name, folder_name):
    
    """
    Checks the existance of the required csv
    If it's not existent and the folder we can create it from is creates the csv
    Asks the user to download the csv if none of the above is true
    """

    if os.path.exists(file_name):
        print('You already have the csv file. Great !')

    elif os.path.exists(folder_name):
        print(
            "You only have the folder :/ This is going to take a long time please download the csv file"
        )
        if (file_name == 'enron_data.csv'):
            create_df(folder_name)
        else :
            create_spam_file(folder_name)

    else:
        print(
            "You don't have the csv file... Please download it using the link above."
        )


####### ####### Data creation ####### #######
def email_analyse(inputfile):
    
    '''
    Returns the email that is in input file in a dataFrame row
    :param inputfile: The path to the file

    :return: array that represents a row in the final dataFrame
    '''
    
    with open(inputfile, "r") as f:
        data = f.read()
        email = Parser().parsestr(data)
    return [
        inputfile[10:], email['Message-ID'], email['Date'], email['From'],
        email['To'], email['Subject'], email['Mime-Version'],
        email['Content-Type'], email['Content-Transfer-Encoding'],
        email['X-From'], email['X-To'], email['X-cc'], email['X-bcc'],
        email['X-Folder'], email['X-Origin'], email['X-FileName'],
        email.get_payload(),
        inputfile.split('/')[2],
        inputfile.split('/')[3]
    ]


def create_df(path):
    
    """
    Starting with the original data_set, ie. a folder with all the subfolders that have all the messages
    Creates a pandas dataframe that has all the information needed for this project
    This takes a very very long time to run so please don't run it unless you don't have the csv file

    :param path: The path to the directory
    """
    
    #initializations
    df = pd.DataFrame(columns=[
        'file', 'Message-ID', 'Date', 'From', 'To', 'Subject', 'Mime-Version',
        'Content-Type', 'Content-Transfer-Encoding', 'X-From', 'X-To', 'X-cc',
        'X-bcc', 'X-Folder', 'X-Origin', 'X-FileName', 'content', 'user',
        'subfolder'
    ])
    print('starting DataFrame creation process ...')
    i = 0
    #traversing through various directories, subdirectories and files
    for directory, subdirectory, filenames in os.walk(path):
        for filename in filenames:
            try:
                df.loc[i] = email_analyse(os.path.join(directory, filename))
            except UnicodeDecodeError:
                continue
            i += 1
            if (i % 1000 == 0):  # Keeps us up to date with the progress
                print(i, 'files processed')
    # Saves the dataFrame in a csv file for future use
    df.to_csv('enron_data.csv')
    print('done, the df is now saved as "enron_data.csv"')


def email_analyse_HAMSPAM(inputfile):
    
    "We consider here that 0 means the e-mail is fine and 1 means it is a spam"
    '''
    Returns the email that is in input file in the spam dataFrame row
    :param inputfile: The path to the file

    :return: array that represents a row in the final spam dataFrame
    '''
    
    with open(inputfile, "r") as f:
        data = f.read()
        email = Parser().parsestr(data)
        filetype = 0
        if inputfile.find('spam') > 1:
            filetype = 1
    return [email['Subject'], email.get_payload(), filetype]


def create_spam_file(path):
    
    """
    Starting with the original data_set, ie. a folder with all the subfolders that have all the messages
    Creates a pandas dataframe that has all the information needed for this project
    This takes a very very long time to run so please don't run it unless you don't have the csv file

    :param path: The path to the directory
    """
    
    SPAM_HAM_df = pd.DataFrame(columns=['Subject', 'Content', 'Spam'])

    i = 0
    for directory, subdirectory, filenames in os.walk(path_):
        for filename in filenames:
            if (filename != 'Summary.txt'):
                try:
                    SPAM_HAM_df.loc[i] = email_analyse_HAMSPAM(
                        os.path.join(directory, filename))
                except UnicodeDecodeError:
                    continue
                i += 1
                if (i % 1000 == 0):
                    print(i)
    SPAM_HAM_df.to_csv('enron_SPAM_HAM.csv')


def add_email_with_pattern(df_financial,data,non_repertorized_emails):
    
    '''
    Returns a dataframe with new emails for employees when finding a match between financial and emails databases
    
    
    :param df_financial: the financial dataframe
     param data: the e-mails dataframe
     param non_repertorized_emails: e-mails of employees we are looking for
     
    :return: dataframe with new emails for employees
    '''
    
    e_mails_found = []
    df=df_financial
    
    for email in non_repertorized_emails:
        index = non_repertorized_emails[non_repertorized_emails == email].index[0]
        l = email.split('@')
        l_ = l[0].split('.')
        fam_name = l_[1]
        giv_name = l_[0]

        potential_emails = list(
            set(data['From'][data['From'].str.contains(fam_name)]))

        for email_potential in potential_emails:
            l_potential = email_potential.split('@')
            l_potential_ = l_potential[0].split('.')
            if (len(l_potential_) > 1):
                fam_name_potential = l_potential_[1]
                giv_name_potential = l_potential_[0]
                if (fam_name_potential == fam_name
                        and giv_name_potential in giv_name):
                    e_mails_found.append(email_potential)
                    df['E-mail'] = df['E-mail'].set_value(
                        index, email_potential)
    return df 

def add_stress_score(df_financial,data,Fear_Keyword):
    
    '''
    Adds a new column to the dataframe containing the number of occurences of fear words in employees e-mails
    
    :param df_financial: The financial dataframe
     param data : The emails dataframe
     param Fear_Keyword : List of the words denoting fear 
     
    :return: dataframe with new stress column
    '''
    
    
    df = df_financial
    EmailsRiseOfFear = data[data['content'].str.contains('|'.join(Fear_Keyword))]
    PeopleWithEmails = df_financial["E-mail"]
    emailsRisingFear_by_person = EmailsRiseOfFear.groupby(
    EmailsRiseOfFear['From'])['user'].count()

    mails = emailsRisingFear_by_person.keys()

    Score_Per_Person = 0
    for i in range(len(PeopleWithEmails)):

        for element in PeopleWithEmails[i]:

            if (element in mails):

                Score_Per_Person += emailsRisingFear_by_person[element]

        df["StressScore"].loc[i] = Score_Per_Person
        Score_Per_Person = 0
        
    return df


def add_emails_df(df,data) :
    
    '''
    Adds new columns to the dataframe containing the number of emails sent and received from employees and POIs
    
    :param df_financial: The financial dataframe
     param data : The emails dataframe 
     
    :return: dataFrame with new columns, Nb_Emails_Sent, Nb_Emails_Received, Nb_Emails_FromPOI, Nb_Emails_ToPOI
    '''
    
    #Mails received by POIs
    mailsToPOI = data[data['To'].str.contains('|'.join(poi_list))]

    #Mails sent by POIs
    mailsFromPOI = data[data['From'].str.contains('|'.join(poi_list))]
    
    df_financial=df
    PeopleWithEmails = df_financial["E-mail"]
    for i in range(len(PeopleWithEmails)):
        sumTo = 0
        sumFrom = 0
        sumFromPOI = 0
        sumToPOI = 0
        for element in PeopleWithEmails[i]:
            sumTo += len(data[data['From'].str.contains(element)])
            sumFrom += len(data[data['To'].str.contains(element)])
            sumFromPOI += len(
                mailsFromPOI[mailsFromPOI['To'].str.contains(element)])
            sumToPOI += len(
                mailsFromPOI[mailsFromPOI['From'].str.contains(element)])

        df_financial["Nb_Emails_Sent"].loc[i] = sumTo
        df_financial["Nb_Emails_Received"].loc[i] = sumFrom
        df_financial["Nb_Emails_FromPOI"].loc[i] = sumFromPOI
        df_financial["Nb_Emails_ToPOI"].loc[i] = sumToPOI
        
    return df_financial


def untraceable_employees(non_repertorized_emails,data): 
    
    '''
    Returns the final list of employees that are not repertorized 
    
    :param non_repertorized_emails: list of employees still unfound
     param data : The emails dataframe 
     
    :return: list of employees that are not repertorized
    '''
    
    l=[]
    for email in non_repertorized_emails:

        found = False

        for To in data['To'].iteritems():

            if (email in To[1]):
                found = True
                break

        if (found == False):
            l.append(email)
    return l

####### ####### Plots ####### #######

def plot_line(size, serie, xlabel, ylabel, title):
    
    '''
    plots a line using the parameters

    :param size: The size of the image
    :param serie: Pandas Series
    :param xlabel: The x label of the plot
    :param ylabel: The y label of the plot
    :param title: The title of the plot
    :param rotaion: The title of the plot

    '''

    plt.figure(figsize=size)
    plt.plot(serie)
    plt.xticks(serie.index)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_barh(size, x, y, color, xlabel, title):
    
    '''
    plots a horizontal bar plot using the parameters

    :param size: The size of the image
    :param x: The labels for the x axis
    :param y: The counts for each label in x
    :param color: The y label of the plot
    :param xlabel: The x label of the plot
    :param title: The title of the plot

    '''
    fig, ax = plt.subplots(figsize=size)
    y_pos = np.arange(len(x))
    count = y

    ax.barh(y_pos, count, align='center', color=color,edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('word count')
    ax.set_title(title)

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def read_list(path):
    
    """
    This function reads the content of a file and returns the lines of the document in a list
    
    :param size: path of the file
    
    :return: list of the lines of the document
    """
    
    f = open(path)
    line = f.readline()
    poi_list = []
    while line:
        email = emailize(line)
        poi_list.append(email)
        line = f.readline()
    return poi_list



####### ####### Text processing ####### #######

def word_count(content, tokenizer):
    
    '''
    Counts the number of occurences of each word in content

    :param content: The text we want to count the number of words in
    :param tokenizer: The tokenizer to use

    :return: A data Frame that has each word and the number of its occurences

    '''

    words = tokenizer.tokenize(content)
    df = pd.DataFrame({'words': words})
    df = df.groupby('words').size()
    df = pd.DataFrame(df)
    df.columns = ['count']
    return df.sort_values(by=['count'], ascending=False)


def text_process(mess):

    '''
    This method removes stop words and punctuation and tokenizes the e-mail into a list of words

    :param mess: The messy text we want to process

    :return: An array of the words processed

    '''
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [
        word for word in nopunc.split()
        if word.lower() not in stopwords.words('english')
    ]

def numerize_column(s):
     
    '''
    This method is to remove the caracters that would not allow us to parse the datafarme columns into values

    :param s: string of characters

    :return: string containing only numbers

    '''
    numerized = s.str.replace('$', '')
    numerized = numerized.str.replace(',', '')
    numerized = numerized.str.replace('(', '')
    numerized = numerized.str.replace(')', '')
    numerized = numerized.str.replace(' ', '')
    numerized = numerized.str.replace('-', '0')
    return numerized

def emailize(s):
    
    '''
    This method returns an e-mail address with the enron format surname.name@enron.com

    :param s: string of characters

    :return: string containing email of employee

    '''
    
    l = s.replace(" ", ",").replace(".", ',').replace('\n', "").split(",")
    l = [i for i in l if len(i) > 2]
    return str(l[1].lower() + '.' + l[0].lower() + '@enron.com')

