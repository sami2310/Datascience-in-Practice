# Title

Investigating the Enron case through the power of data analysis.

# Abstract

In 2000, Enron was one of the largest companies in the United States. Its stock kept soaring higher and higher day by day. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including around 500.000 emails and detailed financial data for top executives.

Getting access to an email database without violating user privacy is close to impossible. The fall of Enron gave us the opportunity to dive into the detailed analysis of emails and how people use them.

During this project, we'll try to investigate this dataset and try to answer the research questions we came up with below.

# Datasets:

We will use the ENRON database from CMU's CS department.
- Its size is 1.82GO. This Database is composed of around 500K e-mails.
- The format of these e-mails is text.
- The DataSet is represented as a list of folders. Each folder contains the MailBox of an employee. Therefore each folder contains a number of subfolders representing for example inbox, outbox...

We will use a subset of the dataset mentioned above available here : [Spam ham dataset](http://www2.aueb.gr/users/ion/data/enron-spam/)
 - Here we have 6 folders of emails in pre-processed form. Each folder has ~5000 emails classified in 2 subfolders (spam and ham).
 - the ham folder will contain emails from the inbox of a user we choose from the big Enron dataset
 - The spam emails are not from the Enron dataset. They have been collected from several spam filtering companies like SpamAssassin and HoneyPot to name a few
 - This dataset is not very big. It will not cause any problems
 - The format of the spam emails is also .txt


Since the scandal, people have actually managed to collect a lot of the company's and employees' financial information that was publicly available.
 - There is a pdf of this information available on findlaw.com. Here is a link to download it :
[pdf download link](https://drive.google.com/open?id=1HdexKNCKnNrm92S2B0Xb9xX45vrGPzgU)
 - People from Udacity managed to compile all of this and provide it as a Python dictionary with each individual as a key and the information about the individual as values. We will convert it to a pandas dataFrame. Here is the link to the data: [Link to the financial data](https://github.com/MayukhSobo/EnronFraud/blob/master/data/final_project_dataset.pkl)

# Data Exploration
  - We found a way to navigate through all the folders and subfolder to get access to the emails (.txt files), read and save them to .csv file. A format we are very familiar with at this point. We did not drop any information by creating this file. Every bit of information could be valuable.
  - Now that the first step is done, we get access to the data as a pandas dataframe. Some of the interesting fields in the dataset are:
    - Date: This will give us an idea on the time periods employees are at their desks writing emails. We could also detect holidays or days off taken by employees.
    - From, To: Most important features for the graphs that we are going to build
    - user: The user whom folder we found the emails in
    - subfolder: inbox , outbox , deleted, spam. Interesting to see what kind of emails people tend to delete or archive. This will help us create a spam detector cf. research questions.
    - subject: the subject of the email
    - content : its content.
  - We will visualize some of the most interesting columns we have discussed above.

# Research questions
  - Can we find the persons of interest (POI) inside a company just by having access to the emails exchanged? The answer to this question should be fairly straightforward : Yes. But can we recreate the hierarchy? This could be more challenging.
  - Can we manage to find the holiday calendar for a company from emails?
  - The U.S. Equal Employment Opportunity Commission, or EEOC, enforces Title VII of the Civil Rights Act of 1964, which was signed by President Lyndon B. Johnson. Title VII prohibits employment discrimination based on race, color, national origin, religion and sex.
    - We will use the names or last names of employees in the email to check if a certain company complies with these rules
    - We will focus on visualizing the racial composition in the institution
  - SPAM detection: Create a machine learning model that will classify emails as spam or ham. This a great opportunity to train such an algorithm since email datasets are very hard to come by. For this, we have found a subset of our dataset that has emails that are already classified. We will use it to train our model.
  - Fraud Detection: We can't talk about Enron without mentioning fraud and manipulation. We have found complementary financial information from findlaw.com. We will train a model to see if we can find individuals who are heavily corrupted.

# Plan of action and ideas on how to implement everything:

 - Exploration:
    -  The time when people send emails: Analyse the date column
    -  People who send most emails: group by user column and aggregate sum the emails sent
    -  What do the emails say?
        - In the subject: split the subject part of the emails and count the appearance frequency of different words.
        - In the content: Do the same for the content
 -  Network Analysis (Everything could be implemented using the networkx python library)
    - Draw (part of) the Network
    - How connected is a given node to all the other nodes?
 - Ethnic Diversity: We will scrape the names of the employees from the e-mail addresses found in the headers of the text files. We will then use the python library ethnicolr that will help us to determine the ethnicity probability distribution of a certain person based on their surname.
 - Spam / Ham Classification: We have done this task many times in similar projects. We will implement different classification algorithms' models and compare their results.
 - Fraud Detection: Same as the previous point but for a different purpose.

# Timeline of the progress:

### December 2nd: Done
 - Do the network analysis part.
 - Work on Ethnic Diversity.
### December 9th: Done
- Continuously enhancing our data exploration/analysis/cleaning.
- Finishing Spam / Ham Classification
- Start working on Fraud Detection

### December 16th: Done 
- Finalize the project and the report.
- Start working on the poster and presentation.
