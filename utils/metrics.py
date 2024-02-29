import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, recall_score, precision_score, confusion_matrix, cohen_kappa_score, accuracy_score

class Estimator():
    def __init__(self, cfg, thresholds=None):
        self.cfg = cfg
        self.thresh = self.cfg.data.threshold
        self.criterion = cfg.train.criterion
        self.num_classes = cfg.data.num_classes
        self.thresholds = [-0.5 + i for i in range(self.num_classes)] if not thresholds else thresholds
        self.bin_thresholds = [-0.5 + i for i in range(2)] if not thresholds else thresholds

        self.reset()  # intitialization

    def reset(self):
        self.correct = 0
        self.bin_correct =0
        self.num_samples = 0
        self.val_loss = []
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

        self.bin_prediction = [] # normal binary task
        self.onset_target = []

        self.y_trues = np.zeros((0), np.int32) # for binary measures
        self.yhat = np.zeros((0), np.int32)

        self.bin_pred = np.zeros((0, 2), np.float32)  # for AUC, ROC, AUPRC
        self.bin_class_pred = np.zeros((0), np.int32) 
        self.bin_class_true = np.zeros((0), np.int32)

    def update(self, predictions, targets):
        targets = targets.data.cpu()

        yprob = F.softmax(predictions.detach().cpu(), dim=-1) # from logit to proba
        yhat = torch.argmax(yprob, dim=1)      # from logit to class predictions

        # save class prediction with true labels
        self.yhat = np.concatenate([self.yhat, yhat.detach().cpu().numpy()])  
        self.y_trues = np.concatenate([self.y_trues, targets])

        predictions = predictions.data.cpu()
        #print('raw prediction: \n', predictions)
            
        if self.cfg.data.binary:
            self.bin_prediction.append(predictions.detach())  
            self.onset_target.append(targets)          
        else:
            y0 = torch.sum(yprob[:, :self.thresh], dim=1, keepdims=True)
            y1 = torch.sum(yprob[:, self.thresh:], dim=1, keepdims=True)
            logits = torch.cat((y0, y1), 1)
            self.bin_pred = np.concatenate([self.bin_pred, logits.numpy()])

            bin_targets = (targets >= self.thresh).long()
            self.bin_class_pred = np.concatenate([self.bin_class_pred, bin_targets.numpy()])

            bin_class_prediction = torch.argmax(logits, dim=1) #(yhat >= self.thresh).long()
            self.bin_class_true = np.concatenate([self.bin_class_true, bin_class_prediction.numpy()])
            
        # update metrics
        predictions = self.to_prediction(predictions) # class prediction: muticlass or not
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        
        for i, p in enumerate(predictions): # muticlass or not
            self.conf_mat[int(targets[i])][int(p.item())] += 1
        
    def get_auc_auprc(self, digits=-1):
        if self.cfg.data.binary:
            y_bin_pred = torch.cat(self.bin_prediction, dim=0) # x axis
            y_onset_target = torch.cat(self.onset_target, dim=0)
            y_pred_proba = torch.nn.functional.softmax(y_bin_pred, 1) # y-axis
            y_pred = torch.argmax(y_pred_proba, dim=1) # class prediction from the model

            y_pred = y_pred.numpy()
            y_pred_proba = y_pred_proba.numpy()
            y_onset_target = y_onset_target.numpy()
        elif self.thresh: 
            y_pred = self.bin_class_true  # class prediction from the model
            y_pred_proba = self.bin_pred
            y_onset_target = self.bin_class_pred # true label
        else:
            raise ValueError('Error => the threshold is not define in the config file')

        cm = confusion_matrix(y_onset_target, y_pred)

        if len(cm)==1:
            if y_onset_target[0]==1:
                rec_score = 1
                prec_score = 1
                specificity_score = 0
            else:
                rec_score = 0
                prec_score = 0
                specificity_score = 1

            list_auc, list_auprc, list_others = [0,0,0], [0,0,0], [0,0,0,0]
        else :  
            if (cm[1,0] + cm[1,1]) !=0:
                rec_score = round(recall_score(y_onset_target, y_pred), digits)
                if cm[0,0]>0:
                    fpr, tpr, thres = roc_curve(y_onset_target, y_pred_proba[:, 1]) # pos_label=1 -> when not 0 and 1
                    bin_auc = auc(fpr, tpr)
                else:
                   fpr, tpr, thres, bin_auc = 0,0,0,0
            else:
                rec_score, fpr, tpr, thres, bin_auc = 0, 0, 0, 0, 0

            if (cm[0,1] + cm[1,1]) !=0:
                prec_score = round(precision_score(y_onset_target, y_pred), digits) 
            else:
                prec_score = 0                

            if (cm[0,0] + cm[0,1]) !=0:
                specificity_score = round(cm[0,0]/(cm[0,0] + cm[0,1]), digits)
            else:
                specificity_score = 0

            if (cm[1,0] + cm[1,1]) !=0:
                if (cm[0,1] + cm[1,1]) !=0:
                    precision, recall, _ = precision_recall_curve(y_onset_target, y_pred_proba[:, 1])
                    au_prc = auc(recall, precision)
                else:
                    precision, recall, au_prc = 0, 0, 0 
            else:
                precision, recall, au_prc = 0, 0, 0     
            
            list_auc = [round(bin_auc, digits), fpr, tpr]
            list_auprc = [round(au_prc, digits), precision, recall]
            list_others = [rec_score, prec_score, specificity_score, cm]

        return list_auc, list_auprc, list_others
            

    def update_val_loss(self, loss):
        self.val_loss.append(loss)

    def get_val_loss(self, digits=-1):
        return round(self.val_loss[-1], digits)

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        if self.cfg.data.threshold:
            bin_acc = accuracy_score(self.bin_class_true, self.bin_class_pred)
            bin_acc = bin_acc if digits == -1 else round(bin_acc, digits)
            return acc, bin_acc
        else: 
            return acc

    def get_cm(self):
        return self.conf_mat

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def get_kappa_v2(self, digits=-1):
        kappa = cohen_kappa_score(self.y_trues, self.yhat, weights='quadratic')
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def to_prediction(self, predictions):
        if self.criterion in ['cross_entropy', 'focal_loss', 'kappa_loss']:
            predictions = torch.tensor(
                [torch.argmax(p) for p in predictions]
            ).long()
        elif self.criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
            predictions = torch.tensor(
                [self.classify(p.item()) for p in predictions]
            ).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i

def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()

    if expected == 1:
        val = (observed - 0.9999) / (1 - 0.9999)
    else:
        val = (observed - expected) / (1 - expected)

    return val
