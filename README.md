**Prediction of Hotel reviews(Good/Bad) from text by natural language processing**

It involves 3 steps 1. Preprocessing of text 2. Bag of words model 3. Classification model
The dataset used contained 1000 reviews which were numbered 1 if review was good and 0 if review was bad.

**Preprocessing or cleaning the text**
First data was separated into 2 columns as data was already separated by text: Reviews(contain text data) and Liked(0 for bad review or 1 for good review). Then Stemming is done to take root of word. For example Loved, Loving will all be represented as love. This helps in dimensionality reduction when we make sparse matrix.

**Bag of words model**
Sparse matrix is created by tokenization. Here for every unique words columns are there, if for all review that word appears 1 is placed in matrix position else 0. In this sparse matrix too many zeros appear hence to reduce sparsity i.e to reducce zeros two methods can be implemented: dimensionality reduction and selecting maximum number of features. Here I have taken 1500 maximum features.

**Classification model**
For natural language processing common machine learning models used are Naive bayes classifier, Decision tree classification and Random forest classification Here I have used Naive based classifier. Testing dataset is 20% of total dataset.
Once prediction is done confusion matrix shows that correct negative predictions were 55 and correct positive prediction were 91 Accuracy=(55+91)/200=0.73. Hence 73% accuracy is obtained.
