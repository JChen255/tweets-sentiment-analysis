import pickle
import pandas as pd
import argparse
from preprocess import preprocess
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix


def load_models(vectorizer,BNB,LR):

    # Load the vectoriser.
    file = open(vectorizer, 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # Load the BNB Model.
    file = open(BNB, 'rb')
    BNBmodel = pickle.load(file)
    file.close()

    # Load the LR Model.
    file = open(LR, 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel, BNBmodel


def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    pred = df['sentiment'].tolist()
    return df,pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectoriser_ngram", type=str, default="apple_vectoriser-ngram.pickle",
                        help = "download the models from pickle files.")
    parser.add_argument("--BNB_Model", type=str, default="apple_BNB.pickle",
                        help = "download the models from pickle files.")
    parser.add_argument("--LR_Model", type=str, default="apple_LR.pickle",
                        help = "download the models from pickle files.")
    parser.add_argument("--putin_test_dataset", type=str, default="Putin tweets.csv",
                        help = "download the testing dataset.")
    args = parser.parse_args()

    # Loading the models from pickle files
    vectorizer = args.vectoriser_ngram
    BNB = args.BNB_Model
    LR = args.LR_Model
    vectoriser, LRmodel, BNBmodel = load_models(vectorizer, BNB, LR)

    # Text to classify should be in a list.
    file = pd.read_csv(args.putin_test_dataset)
    text = file["Text"]
    labels = file["label"]

    # print out predictions for Logistic Regression Model
    pred1 = predict(vectoriser, LRmodel, text)[1]
    print(pred1)
    # print out predictions for Logistic Regression Model
    pred2 = predict(vectoriser, BNBmodel, text)[1]
    print(pred2)

    # Calculate specificity for Logistic Regression Model
    tn1, fp1, fn1, tp1 = confusion_matrix(labels, pred1).ravel()
    specificity1 = tn1 / (tn1 + fp1)
    print("specificity1",specificity1)

    # Calculate specificity for Naive Bayes Model
    tn2, fp2, fn2, tp2 = confusion_matrix(labels, pred2).ravel()
    specificity2 = tn2 / (tn2 + fp2)
    print("specificity2", specificity2)

    # Calculate metrics for Logistic Regression Model
    p1 = precision_score(labels, pred1)
    r1 = recall_score(labels, pred1)
    a1 = accuracy_score(labels, pred1)
    f1 = f1_score(labels, pred1)
    print(f"Sklearn scores for Logistic Regression Model: precision {p1:0.03}\trecall {r1:0.03}\taccuracy{a1:0.03}\tf1{f1:0.03}")

    # Calculate metrics for Naive Bayes Model
    p2 = precision_score(labels, pred2)
    r2 = recall_score(labels, pred2)
    a2 = accuracy_score(labels, pred2)
    f2 = f1_score(labels, pred2)
    print(f"Sklearn scores for BernoulliNB Model: precision {p2:0.03}\trecall {r2:0.03}\taccuracy{a2:0.03}\tf2{f1:0.03}")
