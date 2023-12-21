from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def ROC(y_true, y_score, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - specyficzność')
    plt.ylabel('Czułość')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")
    plt.show()