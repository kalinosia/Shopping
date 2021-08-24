import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def month(given_month):
    if given_month == 'Jan':
        return 0
    elif given_month == 'Feb':
        return 1
    elif given_month == 'Mar':
        return 2
    elif given_month == 'Apr':
        return 3
    elif given_month == 'May':
        return 4
    elif given_month == 'June':
        return 5
    elif given_month == 'Jul':
        return 6
    elif given_month == 'Aug':
        return 7
    elif given_month == 'Sep':
        return 8
    elif given_month == 'Oct':
        return 9
    elif given_month == 'Nov':
        return 10
    elif given_month == 'Dec':
        return 11
    else:
        print('Given month in date is not correct! Month that is not correct: ', given_month)
        raise NameError('Given month in date is not correct!')


def bool_to_int(variable):
    if variable == 'FALSE':
        return 0
    elif variable == 'TRUE':
        return 1
    elif variable == 'Returning_Visitor':
        return 1
    else:
        return 0


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)
    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # (evidence, labels)
    evidence = []
    labels = []

    with open(f"{filename}", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # have row which is dict
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                month(row['Month']),
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                bool_to_int(row['VisitorType']),
                bool_to_int(row['Weekend'])
            ])
            labels.append(bool_to_int(row['Revenue']))

    # print(len(evidence)) 12330
    return (evidence, labels)
    # raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # print(len(labels))
    model = KNeighborsClassifier(n_neighbors=1)
    # C:\Program Files (x86)\Python38-32\lib\site-packages\sklearn\neighbors\_classification.py:179:
    # DataConversionWarning: A column-vector y was passed when a 1d array was expected.
    # Please change the shape of y to (n_samples,), for example using ravel().
    training = model.fit(evidence, np.ravel(labels))

    return model

    # raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).
    Assume each label is either a 1 (positive) or 0 (negative).
    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.
    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # https://sinyi-chou.github.io/classification-auc/

    # Compute how well we performed
    true_positive = 0  # the same and true
    true_negative = 0  # the same and false
    actual_true = 0  # label is true, not matter what is prediction
    actual_false = 0  # -||- false
    # total = 0  # is that matter?? to check only

    for label, prediction in zip(labels, predictions):
        # total += 1
        if label == prediction and label == 1:
            true_positive += 1
        elif label == prediction and label == 0:
            true_negative += 1
        if label == 1:
            actual_true += 1
        elif label == 0:
            actual_false += 1

    sensitivity = true_positive/actual_true
    specificity = true_negative/actual_false

    return sensitivity, specificity
    # raise NotImplementedError


if __name__ == "__main__":
    main()