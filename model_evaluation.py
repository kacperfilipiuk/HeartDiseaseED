from sklearn.metrics import confusion_matrix


def ocen_model_klasyfikacji(y_true, y_pred, digits=2):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    overall_error_rate = 1 - accuracy
    sensitivity = tp / (fn + tp)
    fnr = fn / (fn + tp)
    specificity = tn / (tn + fp)
    fpr = fp / (tn + fp)
    precision = tp / (fp + tp)
    f1 = (2 * sensitivity * precision) / (sensitivity + precision)
    print('Trafność: ', round(accuracy, digits))
    print('Całkowity współczynnik błędu', round(overall_error_rate, digits))
    print('Czułość: ', round(sensitivity, digits))
    print('Wskaźnik fałszywie negatywnych: ', round(fnr, digits))
    print('Specyficzność: ', round(specificity, digits))
    print('Wskaźnik fałszywie pozytywnych: ', round(fpr, digits))
    print('Precyzja: ', round(precision, digits))
    print('Wynik F1: ', round(f1, digits))