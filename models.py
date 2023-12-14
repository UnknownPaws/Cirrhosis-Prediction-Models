import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ohc = True
svm_run = True
dt_run = True
rf_run = True


# Method for printing accuracy, precision, recall, f1 score
def calculate_and_print_metrics(y_true, y_pred, dataset_type):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{dataset_type} Metrics:")
    print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1:.2%}")


cirrhosis_df = pd.read_csv('cirrhosis.csv')

if not ohc:
    categorical_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    X = cirrhosis_df.drop(['Status'] + categorical_features, axis=1)

if ohc:
    # One-hot encode specified categorical features
    categorical_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    X = cirrhosis_df.drop('Status', axis=1)
    # One Hot Encoding
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# y uses lambda function to contain Status within one column as I couldn't get the skleanrn one hot encoding working
y = cirrhosis_df['Status'].apply(lambda x: 1 if x == 'D' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

if svm_run and not ohc:
    # SVM without One-Hot Encoding, without Scaling
    svm_model = SVC(probability=True)
    param_grid = {
        'C': [1],
        'gamma': ['scale'],
        'kernel': ['rbf']
    }
    svm_grid_search = GridSearchCV(svm_model, param_grid, cv=2, scoring='f1')
    svm_grid_search.fit(X_train, y_train)

    best_model_svm = svm_grid_search.best_estimator_
    y_pred_svm_train = best_model_svm.predict(X_train)
    y_pred_svm_test = best_model_svm.predict(X_test)

    calculate_and_print_metrics(y_train, y_pred_svm_train, 'Train - SVM')
    calculate_and_print_metrics(y_test, y_pred_svm_test, 'Test - SVM')

if svm_run and ohc:
    # SVM with One-Hot Encoding, with Scaling
    svm_pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['rbf', 'linear']
    }
    svm_grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='f1')
    svm_grid_search.fit(X_train, y_train)

    best_model_svm = svm_grid_search.best_estimator_
    y_pred_svm_train = best_model_svm.predict(X_train)
    y_pred_svm_test = best_model_svm.predict(X_test)

    calculate_and_print_metrics(y_train, y_pred_svm_train, 'Train - SVM w/ One Hot Encoding & Scaling')
    calculate_and_print_metrics(y_test, y_pred_svm_test, 'Test - SVM w/ One Hot Encoding & Scaling')

if dt_run:
    # Decision Tree
    dt_classifier = DecisionTreeClassifier(random_state=1)
    dt_classifier.fit(X_train, y_train)
    y_pred_dt_train = dt_classifier.predict(X_train)
    y_pred_dt_test = dt_classifier.predict(X_test)
    calculate_and_print_metrics(y_train, y_pred_dt_train, 'Train - Decision Tree')
    calculate_and_print_metrics(y_test, y_pred_dt_test, 'Test - Decision Tree')

if rf_run:
    # Random Forest
    rf_classifier = RandomForestClassifier(random_state=1)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf_train = rf_classifier.predict(X_train)
    y_pred_rf_test = rf_classifier.predict(X_test)
    calculate_and_print_metrics(y_train, y_pred_rf_train, 'Train - Random Forest')
    calculate_and_print_metrics(y_test, y_pred_rf_test, 'Test - Random Forest')
