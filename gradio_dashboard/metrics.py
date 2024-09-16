import pathlib

import joblib as jb
import pandas as pd
import plotly.express as px
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)

# load data
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
test_data = pd.read_csv(DATA_PATH.joinpath("test_data.csv"))

# split data
target = "Exited"
X = test_data.drop(columns=["RowNumber", "Exited"])
y = test_data[target]

# load model
MODEL_PATH = PATH.joinpath("model").resolve()
model = jb.load(MODEL_PATH.joinpath("model.pkl"))

# get prediction probabilities
prediction_probabilites = model.predict_proba(X)[:,1]

# define metrics function
def show_metrics(probability_threshold):
    """
    parameters
    ----------
    probability_threshold: float
        probability threshold for model prediction
    
    return
    -------
    figure: ConfusionMatrix
        Confusion matrix showing True values vs Predicted values
    evaluation metrics: float
        Accuracy, Precision, and Recall scores
    """

    # get predictions
    predictions = (prediction_probabilites >= probability_threshold).astype(int)

    # get confusion matrix
    matrix_array = confusion_matrix(y, predictions)

    # plot confusion matrix
    fig = px.imshow(
        matrix_array,
        labels=dict(x="Predicted Value", y="True Value"),
        x = ["False", "True"],
        y = ["False", "True"],
        title="Confusion Matrix",
        text_auto=True,
        aspect="auto",
        color_continuous_scale=px.colors.sequential.gray
        )
    # fig.update_xaxes(side="bottom")
    fig.update_layout(title_x=0.5, coloraxis_showscale=False)

    # get metrics
    accuracy = round(accuracy_score(y, predictions), 2)
    precision = round(precision_score(y, predictions), 2)
    recall = round(recall_score(y, predictions), 2)

    return fig, accuracy, precision, recall

def estimate_costs(probability_threshold, crc, cac):
    """
    parameters
    ----------
    probability_threshold: float
        probability threshold for model prediction
    crc: int
        Customer Retention Cost
    cac: int
        Customer Aquisition Cost

    return
    -------
    retention_cost: int
        Total amount spent on customer retention
    aqusition_cost: int
        Total amount sent on customer acqusition
    total_amount: int
        Total cost of customer retention and aquisition
    amount_saved: int
        amount saved due to model precision
    """

    # get predictions
    predictions = (prediction_probabilites >= probability_threshold).astype(int)

    # get confusion matrix
    matrix_array = confusion_matrix(y, predictions)

    # get costs
    retention_cost = matrix_array.sum(axis=0)[1] * crc # crc cost with model
    aquisition_cost = matrix_array[1][0] * cac # cac with model
    total_amount = retention_cost + aquisition_cost # total_amount amount spent on customer aquisition and retention
    amount_saved = (matrix_array.sum(axis=1)[1] * cac) - total_amount

    return retention_cost, aquisition_cost, total_amount, amount_saved