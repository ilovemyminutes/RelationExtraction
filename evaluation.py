from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate(y_true, y_pred, average: str = "macro") -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=average
    )
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)
