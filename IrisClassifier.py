# IRIS CLASSIFICATION MODEL WITH SCIKIT


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# prepare the dataset by splitting into feature array with corresponding label array
features = iris.data
labels = iris.target

# split data into training and testing data - note: the data is shuffled and then split between arrays
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.5)

my_classifier = KNeighborsClassifier()

# pass training data into classifier
my_classifier.fit(features_train, labels_train)

# pass test data into model and store predictions in variable so we can test accuracy later
prediction = my_classifier.predict(features_test)

# prints accuracy of predicted labels compared to true labels
## print(accuracy_score(labels_test, prediction))

# make up a flower that is similar to a datapoint to test if it correctly predicts the type for data it hasn't seen before
iris1 = [[6.8,3.0,5.5,2.2]] 

prediction1 = my_classifier.predict(iris1)

if prediction1 == 0:
    print('setosa')
if prediction1 == 1:
    print('versicolor')
if prediction1 == 2:
    print('virginica')


# KNEIGHBOURS 