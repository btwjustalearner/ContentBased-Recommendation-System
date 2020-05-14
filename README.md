# ContentBased-Recommendation-System

In this task, I build a content-based recommendation system by generating profiles from review texts for users and businesses in the train review set. Then you will use the system/model to predict if a user prefers to review a given business, i.e., computing the cosine similarity between the user and item profile vectors.

During the training process, you will construct business and user profiles as the model:

a. Concatenating all the review texts for the business as the document and parsing the document, such as removing the punctuations, numbers, and stopwords. Also, you can remove extremely rare words to reduce the vocabulary size, i.e., the count is less than 0.0001% of the total words.

b. Measuring word importance using TF-IDF, i.e., term frequency * inverse doc frequency

c. Using top 200 words with highest TF-IDF scores to describe the document

d. Creating a Boolean vector with these significant words as the business profile

e. Creating a Boolean vector for representing the user profile by aggregating the profiles of the items that the user has reviewed

During the predicting process, you will estimate if a user would prefer to review a business by computing the cosine distance between the profile vectors. The (user, business) pair will be considered as a valid pair if their cosine similarity is >= 0.01. You should only output these valid pairs.
