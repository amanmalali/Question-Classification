# Question Classification

### Prerequisites

Pandas
Numpy
SciKitLearn
Natural Language Toolkit(NLTK)
Written and compiled on the Jupyter Notebook

### Preprocessing of data

The text document of the questions was first split according to its constituent
sentences. These sentences were further split into its constituent words and then the
question number any punctuation was also removed inorder to standardize the
questions. The numbers in the questions itself were untouched because some of them
included years such as “1680” and these contributed classification of the sentences in
most cases. After this the words were reconstituted into a sentence and placed into a
list of strings.

### Creating Features

As the classifier is based on the bag of words model, this requires a numeric
representation of the frequency of every word in all the questions. Bag of words is a
rudimentary approach to text classification in this case question classification but
because of its robust nature and its ability to work well smaller sized data made it the
perfect choice for the model for this particular task.
For this the questions were then passed to the CountVectorizer function which split
them into its corresponding matrix of token counts. This gives us a representation of
the frequency of all the words that occur in all the questions and thus gives us a
representation of what kind of words would put a particular question under a
particular category or label.
None of the features were removed due to the fact that most words occurred in more
than 90% of the documents and losing these features would drastically affect the
output of the classifier. Considering also the small size of the training data any loss of
features would affect the classifier performance.The training data was split into a
90:10 ratio for the actual training data and the testing data.

### The Classifier

A Linear Support Vector Classification was used as a classifier model for this task.
This choice was made after attempts with SVM as well as Random Forest Classifier.
The ability of LinearSVC to handle multi label data by using the one-vs-rest scheme.
Also known as one-vs-all, this strategy consists of fitting one classifier per class or
label. For each classifier, the class is fitted against all the other classes. In addition
to its computational efficiency, another advantage of this approach is interpretability.
Since each class is represented by one and one classifier only, it is possible to gain
knowledge about the class by inspecting its corresponding classifier. This solution
works particularly well in this case because of the multilabel nature of the
classification of questions

## Authors

* **Aman Malali** 
