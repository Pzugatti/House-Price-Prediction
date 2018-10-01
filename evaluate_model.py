from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from sklearn import metrics


class clf_eval:

    def __init__(self):
        self.message = 'Running'

    def __repr__(self):
        return self.message

    @staticmethod
    def plot_loss_acc_curves(history):
        """Plot the loss and accuracy curves for training and validation data"""
        fig, ax = plt.subplots(2,1)
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def Clf_report(model,Ytrue,YPred):
        print("Classification report for classifier %s:\n%s\n"
              % (model, metrics.classification_report(Ytrue, YPred)))

class reg_eval:

    @staticmethod
    def adjusted_r2(r2,x,y):
        return 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)

    @staticmethod
    def residual_plot(actual_y,predicted_y):
        res_plot = plt.figure(1)
        res_plot.set_figheight(8)
        res_plot.set_figwidth(12)
        res_plot.axes[0] = sns.residplot(predicted_y, actual_y,
                                  lowess=True,
                                  scatter_kws={'alpha': 0.5},
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        res_plot.axes[0].set_title('Residuals vs Fitted')
        res_plot.axes[0].set_xlabel('Fitted values')
        res_plot.axes[0].set_ylabel('Residuals')
        plt.show()

    @staticmethod
    def qq_plot(residual_norm):
        QQ = ProbPlot(residual_norm)
        res_plot = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
        res_plot.set_figheight(8)
        res_plot.set_figwidth(12)
        res_plot.axes[0].set_title('Normal Q-Q')
        res_plot.axes[0].set_xlabel('Theoretical Quantiles')
        res_plot.axes[0].set_ylabel('Standardized Residuals')
        plt.show()
