import torch.nn.functional as F
import torch.nn as nn
import torch
class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True, reduction='batchmean')):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
        
class CELoss(nn.Module):
    """Cross Entropy Loss for contrastive learning.
    args:
        error_metric:  What base loss to use (CrossEntropyLoss by default).
    """

    def __init__(self, error_metric=nn.CrossEntropyLoss()):
        super().__init__()
        print('=========using Cross Entropy Loss==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        # CrossEntropyLoss in PyTorch expects labels to be class indices
        # So, we get the class indices from the one-hot encoded labels
        _, label_indices = torch.max(label, 1)
        loss = self.error_metric(prediction, label_indices)
        return loss