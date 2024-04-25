import datetime
import logging
import os

import numpy as np
from nodule_classifier.dsetsG import  LunaDataset_Train, LunaDataset_Val
from nodule_classifier.modelG import LunaModel
import torch
from  torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from util.utilG import enumerateWithEstimate

log = logging.getLogger('TrainingLoop')
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)
Y_TRUE = 0
Y_PRED = 1
LOSS = 2
class TrainingLoop:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.trn_writer = None
        self.val_writer = None
        
    
    def run(self, epochs=1):
        
        self.model.to(self.device)
            
        for epoch_idx in range(1, epochs + 1):
            
            #training
            epoch_metrics_tensor_train = torch.zeros(3, len(self.train_loader.dataset)) 
            # we want the number of samples, not the number of batches
            self.model.train()
            for batch in enumerateWithEstimate(self.train_loader, f"EPOCH_train: {epoch_idx}"):
                self.optimizer.zero_grad()
                loss_scalar = self.compute_batch_loss_and_populate_metrics(batch, epoch_metrics_tensor_train)
                loss_scalar.backward()
                self.optimizer.step()
            self.logMetrics(epoch_idx, 'trn', epoch_metrics_tensor_train)
            
            #validation
            epoch_metrics_tensor_val = torch.zeros(3, len(self.val_loader.dataset))
            self.model.eval()
            for batch in enumerateWithEstimate(self.val_loader, f"EPOCH_val: {epoch_idx}"):
                with torch.no_grad():
                    self.compute_batch_loss_and_populate_metrics(batch, epoch_metrics_tensor_val)
            self.logMetrics(epoch_idx, 'val', epoch_metrics_tensor_val)             
                    
        if hasattr(self, 'trn_writer'):
            assert self.trn_writer and self.val_writer
            self.trn_writer.close()
            self.val_writer.close()
    
    def compute_batch_loss_and_populate_metrics(self, batch, epoch_metrics_tensor):
        batch_idx, batch_data = batch
        inputs, y_true = batch_data
        
        y_pred_logits, y_pred_prob = self.model(inputs.to(self.device)) 
        # loss_tensor = self.loss_fn(y_pred_logits, y_true[:, 1]) #could have used nn.NLLLoss() 
        loss_tensor = self.loss_fn(y_pred_logits, y_true.to(self.device)) #could have used nn.NLLLoss() 
        
        slice_start = batch_idx * self.train_loader.batch_size
        slice_end = slice_start + len(y_true)
        epoch_metrics_tensor[LOSS, slice_start:slice_end] = loss_tensor.detach()
        # epoch_metrics_tensor[Y_TRUE, slice_start:slice_end] = y_true[:, 1].detach()
        epoch_metrics_tensor[Y_TRUE, slice_start:slice_end] = y_true.detach()
        epoch_metrics_tensor[Y_PRED, slice_start:slice_end] = y_pred_prob[:, 1].detach()
        
        return loss_tensor.mean() # reduce to scalar
    
    def logMetrics( # log metrics for each epoch
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', 'Nodule_Classifier', 
                                   datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + 'Nodule_Classifier')
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + 'Nodule_Classifier')
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[Y_TRUE] <= classificationThreshold #groundtruth negative
        negPred_mask = metrics_t[Y_PRED] <= classificationThreshold #prediction negative

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = trueNeg = int((negLabel_mask & negPred_mask).sum()) #true negatives
        pos_correct = truePos = int((posLabel_mask & posPred_mask).sum()) #true positives

        falsePos = neg_count - trueNeg
        falseNeg = pos_count - truePos
        
        metrics_dict = {}
        metrics_dict['loss_all'] = \
            metrics_t[LOSS].mean() #avg loss per epoch of all batches
        metrics_dict['loss_negClass'] = \
            metrics_t[LOSS, negLabel_mask].mean() #avg loss per epoch of negative class of all batches
        metrics_dict['loss_posClass'] = \
            metrics_t[LOSS, posLabel_mask].mean() #avg loss per epoch of positive class of all batches

        # metrics_dict['correct/all'] = (pos_correct + neg_correct) \
        #     / np.float32(metrics_t.shape[1]) * 100 # all correct predictions / all predictions
        metrics_dict['correct_all_accuracy'] = (truePos + trueNeg) / np.float32(pos_count + neg_count) * 100
        # metrics_dict['correct/neg'] = trueNeg / np.float32(neg_count) * 100 # true negatives / all actual negatives = specificity = true nagative rate
        metrics_dict['specificity'] = trueNeg / np.float32(trueNeg + falsePos) * 100 #true nagative rate
        # metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100 # true positives / all actual positives = recall = true positive rate
        metrics_dict['recall'] = truePos / np.float32(truePos + falseNeg) 
        metrics_dict['recall_pct'] = metrics_dict['recall'] * 100
    
        metrics_dict['precision'] = truePos / np.float32(truePos + falsePos) 
        metrics_dict['f1_score'] = (2 * (metrics_dict['precision'] * metrics_dict['recall']) 
                                    / np.float32(metrics_dict['precision'] + metrics_dict['recall']))
        
        
        log.info(
            ("E{} {:8} {loss_all:.4f} loss, "
                  "{correct_all_accuracy:-5.1f}% accuracy, "
                  "{recall:.4f} recall, "
                  "{precision:.4f} precision, "
                    "{f1_score:.4f} f1_score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss_negClass:.4f} loss, "
                "{specificity:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=trueNeg,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss_posClass:.4f} loss, "
                "{recall_pct:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=truePos,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer:SummaryWriter = getattr(self, mode_str + '_writer')
         
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, epoch_ndx)

        writer.add_pr_curve(
            'pr',
            metrics_t[Y_TRUE],
            metrics_t[Y_PRED],
            epoch_ndx
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[Y_PRED] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[Y_PRED] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[Y_PRED, negHist_mask],
                epoch_ndx,
                bins=bins, # type: ignore
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[Y_PRED, posHist_mask],
                epoch_ndx,
                bins=bins, # type: ignore
            )
            
def build_training_loop():
    unzipped_path = 'D:/LIDC-IDRI_unzipped'
    model = LunaModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    batch_size = 64
    num_workers = 4
    # dataset = LunaDataset()
    trainset = LunaDataset_Train()
    valset = LunaDataset_Val()
    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers)
    return TrainingLoop(model, optimizer, loss_fn, train_loader, val_loader)
