import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from everywhereml.sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model(X, y, n_estimators=10):
    """
    Test the model with a random forest classifier
    :param X:
    :param y:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    clf = RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(clf.cross_val_score(X_test, y_test, cv=5))
    y_pred = clf.predict(X_test)
    target_names = ['0', '1', '2']
    print(classification_report(y_test, y_pred, target_names=target_names))
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    # plt.show()
    

def get_arduino_code(X, y, n_estimators=10):
    clf = RandomForestClassifier(n_estimators=n_estimators).fit(X, y)
    print(clf.to_arduino(instance_name='blowClassifier'))

def load_data_from_csv(filename: str, label_column: str) -> tuple:
    """
    Convert csv file to X and y
    :param label_column:
    :param filename:
    :return:
    """
    df = pd.read_csv(filename)
    x_columns = [c for c in df.columns if c != label_column]
    X_pre = df[x_columns]
    for r in X_pre.itertuples():
        if "nope" in r:
            print('Error: row has missing values', r)
    X = df[x_columns].to_numpy(dtype=float)
    y_string = df[label_column]
    label_encoder = LabelEncoder().fit(y_string)
    y_numeric = label_encoder.transform(y_string)
    print('Label mapping', {label: i for i, label in enumerate(label_encoder.classes_)})

    return X, y_numeric


X, y = load_data_from_csv('training-data-4.csv', label_column='label')
# for n in ["training-data-3-1.csv", "training-data-3-2-more-balanced.csv", "training-data-3-3-more-balanced.csv", "training-data-3-4.csv", "training-data-3-5.csv", "training-data-3-6.csv", "training-data-4.csv", "training-data-4-cleaned.csv", "training-data-4-1.csv"]:
# for n in ["training-data-4.csv", "training-data-4-cleaned.csv", "training-data-4-1.csv"]:
#     print(f"######## Testing {n}")
#     X, y = load_data_from_csv(n, label_column='label')
#     for i in [1, 5, 10, 20]:
#         print(f"######## Testing {i} estimators")
#         test_model(X, y, n_estimators=i)
    # test_model(X, y, n_estimators=5)
X, y = load_data_from_csv('training-data/training-data-4-cleaned.csv', label_column='label')
test_model(X, y, n_estimators=10)
get_arduino_code(X, y, n_estimators=10)