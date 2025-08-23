import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Callable, List

def compute_loss_metrics(metric: str, y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute the loss metrics
    current options: mse, mae, rmse, r2
    """
    if metric == 'mse':
        return torch.mean((y_true - y_pred) ** 2)
    elif metric == 'mae':
        return torch.mean(torch.abs(y_true - y_pred))
    elif metric == 'rmse':
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    elif metric == 'r2':
        return 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)
    #input the logits for below:
    elif metric == 'binary_acc':
        # yp = torch.nn.functional.sigmoid(y_pred).round()
        yp = (1 + (y_pred).sign())//2
        return 1/len(yp)*(yp == y_true).sum()
    elif metric == 'cross_acc':
        yp = y_pred.argmax(axis=1)
        return 1/len(yp)*(yp == y_true).sum()
    else:
        raise ValueError('Metric not implemented')

class Metrics(nn.Module):
    """
    Defines and calculate  metrics to be logged.
    """

    def __init__(
        self,
        target_name: str,
        predict_name: Optional[str] = None,
        output_index: Optional[int] = None, # used for multi-task learning
        name: Optional[str] = None,
        metric_keys: List[str] = ["mae"],
        per_atom: bool = False,
        avge0 = 0,
        sigma = 1,
    ):
        """
        Args:
        target_name: name of the target in the dataset
        predict_name: name of the prediction in the model output
        name: name of the metrics
        metric_keys: list of metrics to be calculated
        per_atom: whether to calculate the metrics per atom
        """

        super().__init__()
        self.target_name = target_name
        self.predict_name = predict_name or target_name
        self.output_index = output_index
        self.name = name or target_name
        self.avge0 = avge0
        self.sigma = sigma

        self.per_atom = per_atom

        self.metric_keys = metric_keys
        self.logs = {
            "train": {'pred': [], 'target': []},
            "val": {'pred': [], 'target': []},
            "test": {'pred': [], 'target': []},
        }

    def _collect_tensor(self,
                pred: Dict[str, torch.Tensor],
                target: Optional[Dict[str, torch.Tensor]] = None,
               ):
        pred_tensor = pred[self.predict_name].clone().detach()
        if self.output_index is not None:
            pred_tensor = pred_tensor[..., self.output_index]
        pred_tensor = (pred_tensor*self.sigma) + self.avge0 #adjust for forces?
        if target is not None:
            target_tensor = target[self.target_name].clone().detach()
        elif self.predict_name != self.target_name:
            target_tensor = pred[self.target_name].clone().detach()
        else:
            raise ValueError("Target is None and predict_name is not equal to target_name")

        if self.per_atom:
            n_atoms = torch.bincount(target['batch']).clone().detach()
            pred_tensor = pred_tensor / n_atoms
            target_tensor = target_tensor / n_atoms
        return pred_tensor, target_tensor

    def forward(self, 
                pred: Dict[str, torch.Tensor],
                target: Optional[Dict[str, torch.Tensor]] = None,
               ):
        pred_tensor, target_tensor = self._collect_tensor(pred, target)
        metrics_now = {}
        for metric in self.metric_keys:
            metrics_now[metric] = compute_loss_metrics(metric, target_tensor, pred_tensor)

        return metrics_now

    def update_metrics(self, subset: str, 
                       pred: Dict[str, torch.Tensor], 
                       target: Optional[Dict[str, torch.Tensor]] = None,
                      ):
        pred_tensor, target_tensor = self._collect_tensor(pred, target)
        self.logs[subset]['pred'].append(pred_tensor)
        self.logs[subset]['target'].append(target_tensor)

    def retrieve_metrics(self, subset: str, clear: bool = True, print_log: bool = True):
        pred_tensor = torch.cat(self.logs[subset]['pred'], dim=0)
        target_tensor = torch.cat(self.logs[subset]['target'], dim=0)

        assert pred_tensor.shape == target_tensor.shape, f"pred_tensor.shape: {pred_tensor.shape}, target_tensor.shape: {target_tensor.shape}"

        if pred_tensor.shape[0] == 0:
            raise ValueError("No data in the logs")

        metrics_now = {}
        for metric in self.metric_keys:
            metric_mean = compute_loss_metrics(metric, target_tensor, pred_tensor)
            metrics_now[metric] = metric_mean
            if print_log:
                print(
                    f'{subset}_{self.name}_{metric}: {metric_mean:.6f}',
                )
            logging.info(
                f'{subset}_{self.name}_{metric}: {metric_mean:.6f}',
            )
        if clear:
            self.clear_metrics(subset)

        return metrics_now

    def clear_metrics(self, subset: str):
        self.logs[subset]['pred'] = []
        self.logs[subset]['target'] = []

    def __repr__(self):
        return f'{self.__class__.__name__} name: {self.name}, metric_keys: {self.metric_keys}'


class GetLoss(nn.Module):
    """
    Defines mappings to a loss function and weight for training
    """

    def __init__(
        self,
        target_name: str,
        predict_name: Optional[str] = None,
        output_index: Optional[int] = None, # only used for multi-output models
        name: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: Union[float, Callable] = 1.0, # Union[float, Callable] means that the type can be either float or callable
    ):
        """
        Args:
            target_name: Name of target in training batch.
            name: name of the loss object
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
                         This can be a float or a callable that takes in the loss_weight_args
                         For example, if we want the loss weight to be dependent on the epoch number
                         if training == True and a default value of 1.0 otherwise,
                         loss_weight can be, e.g., lambda training, epoch: 1.0 if not training else epoch / 100
        """
        super().__init__()
        self.target_name = target_name
        self.predict_name = predict_name or target_name
        self.output_index = output_index
        self.name = name or target_name
        self.loss_fn = loss_fn
        # the loss_weight can either be a float or a callable        
        self.loss_weight = loss_weight

    def forward(self, 
                pred: Dict[str, torch.Tensor], 
                target: Optional[Dict[str, torch.Tensor]] = None,
                loss_args: Optional[Dict[str, torch.Tensor]] = None
                ):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        if isinstance(self.loss_weight, Callable):
            if loss_args is None:
                loss_weight = self.loss_weight()
            else:
                loss_weight = self.loss_weight(**loss_args)
        else: 
            loss_weight = self.loss_weight

        if self.output_index is None:
            pred_tensor = pred[self.predict_name]
        else:
            pred_tensor = pred[self.predict_name][..., self.output_index]

        # if pred_tensor.shape != target[self.target_name].shape:
        #     pred_tensor = pred_tensor.reshape(target[self.target_name].shape)

        if target is not None:
            target_tensor = target[self.target_name]
        elif self.predict_name != self.target_name:
            target_tensor = pred[self.target_name]
        else:
            raise ValueError("Target is None and predict_name is not equal to target_name")

        loss = loss_weight * self.loss_fn(pred_tensor, target_tensor)
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, loss_fn={self.loss_fn}, loss_weight={self.loss_weight})"
            )