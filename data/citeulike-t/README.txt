This dataset, citeulike-t, was used in the paper 'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li]. It was collected from CiteULike and Google Scholar. CiteULike allows users to create their own collections of articles. There are abstracts, titles, and tags for each article. Other information like authors, groups, posting time, and keywords is not used in this paper 'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li]. The details can be found at http://www.citeulike.ort/faq/data.adp. 

Processing:
----------
The text information (item content) of citeulike-a is preprocessed by following the same procedure as that in citeulike-a. After removing the stop words, we choose the top 20000 discriminative words according to the tf-idf values as our vocabulary.

Some statistics are listed as follows:

#users 					7947 
#items 					25975 
#tags 					52946 

DATA FILES
mult.dat		bag of words/TF-IDF for each article
raw-data.csv	        raw data
tags.dat		tags, sorted by tag-id's
users.dat		rating matrix (user-item matrix)
vocabulary.dat	        corresponding words for file mult.dat

2 directories with different settings are created P3,P1

P10/
train_P10_1.dat | train_P10_2.dat | train_P10_3.dat
test_P10_1.dat  | test_P10_2.dat  | test_P10_3.dat

P5/
train_P5_1.dat | train_P5_2.dat | train_P5_3.dat
test_P5_1.dat  | test_P5_2.dat  | test_P5_3.dat

P1/
train_P1_1.dat | train_P1_2.dat | train_P1_3.dat
test_P1_1.dat  | test_P1_2.dat  | test_P1_3.dat
