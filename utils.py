import torch
import logging

class Controller:
    def __init__(self, early_stop_patience):
        self.original_early_stop_patience_value = early_stop_patience
        self.early_stop_patience = early_stop_patience
        self.state_dict = {
            # `step` is for training stage
            'train_step': 0,
            # below are for validation stage
            'val_step': 0,
            'epoch': 0,
            'best_imputation_MAE': 1e9,
            'should_stop': False,
            'save_model': True
        }

    def epoch_num_plus_1(self):
        self.state_dict['epoch'] += 1

    def __call__(self, stage, info=None, logger=None):
        if stage == 'train':
            self.state_dict['train_step'] += 1
        else:
            self.state_dict['val_step'] += 1
            self.state_dict['save_model'] = False
            current_imputation_MAE = info['imputation_MAE']
            imputation_MAE_dropped = False  # flags to decrease early stopping patience

            # update best_loss
            if current_imputation_MAE < self.state_dict['best_imputation_MAE']:
                self.state_dict['best_imputation_MAE'] = current_imputation_MAE
                imputation_MAE_dropped = True
            if imputation_MAE_dropped:
                self.state_dict['save_model'] = True

            if self.state_dict['save_model']:
                self.early_stop_patience = self.original_early_stop_patience_value
            else:
                # if use early_stopping, then update its patience
                if self.early_stop_patience > 0:
                    self.early_stop_patience -= 1
                elif self.early_stop_patience == 0:
                    logger.info('early_stop_patience has been exhausted, stop training now')
                    self.state_dict['should_stop'] = True  # to stop training process
                else:
                    pass  # which means early_stop_patience_value is set as -1, not work

        return self.state_dict

# logger
def setup_logger(log_file_path, log_name, mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False  # prevent the child logger from propagating log to the root logger (twice), not necessary
    return logger

# metric
def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def masked_mse_cal(inputs, target, mask):
	""" calculate Mean Square Error"""
	return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """ calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """ calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)


