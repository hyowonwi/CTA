import os
import torch
import traceback

import numpy as np
from absl import flags, app
from datetime import datetime, timedelta
from ml_collections.config_flags import config_flags

import utils
from make_model import make_model
from dataset.unified_dataloader import UnifiedDataLoader

import time 

RANDOM_SEED = 2022
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)

torch.autograd.set_detect_anomaly(True)

os.makedirs('./logs', exist_ok=True)
log_file_path = "./logs/test_{}.log".format((datetime.now()).strftime('%m%d_%H%M'))
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("log_path", log_file_path, "Log File Path.")
flags.DEFINE_string("mode", "train", "Select [ train / test ]")
flags.mark_flags_as_required(["config"])
FLAGS = flags.FLAGS


def test(model, test_dataloader, logger, device, return_metric=False):
    with torch.no_grad():
        
        predictions_collector = []
        ground_truth_collector = []
        entire_mask_collector = []

        for i, batch in enumerate(test_dataloader):
            _, X_doubledot, observed_mask, X, removed_mask, coeffs = map(lambda x: x.to(device), batch)
            preds, pred_final, kld_loss = model(coeffs, X_doubledot, observed_mask, is_test=True)

            predictions_collector.append(pred_final)
            ground_truth_collector.append(X)
            entire_mask_collector.append(removed_mask)

        predictions = torch.cat(predictions_collector)
        ground_truth = torch.cat(ground_truth_collector)
        entire_mask = torch.cat(entire_mask_collector)

        # SAITS metric
        mae = utils.masked_mae_cal(predictions, ground_truth, entire_mask)
        rmse = utils.masked_rmse_cal(predictions, ground_truth, entire_mask)
        mre = utils.masked_mre_cal(predictions, ground_truth, entire_mask)
        mse = utils.masked_mse_cal(predictions, ground_truth, entire_mask)

        if return_metric:
            return {'MAE': mae.item(), 'RMSE': rmse.item(), 'MRE': mre.item(), 'MSE': mse.item()}
        else:
            logger.info('=> MAE: {:.3f}   RMSE: {:.3f}   MRE: {:.3f}   MSE: {:.3f}'.format(mae, rmse, mre, mse))
            
def validation(model, val_dataloader, training_controller, logger, device):
    predictions_collector = []
    ground_truth_collector = []
    entire_mask_collector = []

    # logger.info('computing metrics...')
    for i, batch in enumerate(val_dataloader):
        _, X_doubledot, observed_mask, X, removed_mask, coeffs = map(lambda x: x.to(device), batch)
        preds, pred_final,  kld_loss = model(coeffs, X_doubledot, observed_mask, is_test=True)

        predictions_collector.append(pred_final)
        ground_truth_collector.append(X)
        entire_mask_collector.append(removed_mask)

    predictions = torch.cat(predictions_collector)
    ground_truth = torch.cat(ground_truth_collector)
    entire_mask = torch.cat(entire_mask_collector)

    # SAITS metric
    mae = utils.masked_mae_cal(predictions, ground_truth, entire_mask)

    info_dict = { 'imputation_MAE': mae }
    state_dict = training_controller('val', info_dict, logger)

    return state_dict

def train(model, train_dataloader, val_dataloader, logger, config, device, test_dataloader):
    logger.info("Start Training")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr) 
    training_controller = utils.Controller(config.training.patience)

    for epoch in range(1, config.training.epoch+1):

        model.train()
        recon1_error, recon2_error, kld_error, batch_len = [], [], [], []

        for i, batch in enumerate(train_dataloader):

            mit = config.training.masked_imputation_task
            if mit:
                _, X_doubledot, observed_mask, X, removed_mask, coeffs = map(lambda x: x.to(device), batch)
                preds, pred_final, kld_loss = model(coeffs, X_doubledot, observed_mask, is_test=False)

                reconstruction1 = utils.masked_mae_cal(pred_final, X, observed_mask)
                for pred in preds:
                    reconstruction1 += utils.masked_mae_cal(pred, X, observed_mask)
                reconstruction1 /= len(preds) + 1

                reconstruction2 = utils.masked_mae_cal(pred_final, X,removed_mask)
                
            else:
                _, X, observed_mask, coeffs, deltas = map(lambda x: x.to(device), batch)
                preds, pred_final, kld_loss = model(coeffs, X, observed_mask, deltas)

                reconstruction1 = utils.masked_mae_cal(pred_final, X, observed_mask)
                for pred in preds:
                    reconstruction1 += utils.masked_mae_cal(pred, X, observed_mask)
                reconstruction1 /= len(preds) + 1

                reconstruction2 = torch.zeros_like(reconstruction1)

            kld_regularization = torch.mean(kld_loss[torch.isfinite(kld_loss)])

            loss = reconstruction1 + reconstruction2 + config.training.kld_weight * kld_regularization

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()
            
            recon1_error.append(reconstruction1.item())
            recon2_error.append(reconstruction2.item())
            kld_error.append(kld_regularization.item())
            batch_len.append(len(X))
        
        # lr_scheduler.step() # you can set it like this!

        batch_len = np.array(batch_len) / sum(batch_len)
        recon1_error = np.sum(np.array(recon1_error) * batch_len)
        recon2_error = np.sum(np.array(recon2_error) * batch_len)
        kld_error = np.sum(np.array(kld_error) * batch_len)

        training_loss = recon1_error + recon2_error + config.training.kld_weight * kld_error
 

        logger.info('Epoch: {}  Training loss: {:.3f} <- MAE for observed({:.3f}) + MAE for intentionally removed({:.3f}) + {} * KLD({:.4f})'.format( \
                                                    epoch, training_loss, recon1_error, recon2_error, config.training.kld_weight, kld_error))    
        
        # validation
        model.eval()

        state_dict = validation(model, val_dataloader, training_controller, logger, device)
        early_stopping = state_dict['should_stop']

        if state_dict['save_model']:
            logger.info('Best validation MAE is updated: {:.3f} at epoch {}'.format(state_dict['best_imputation_MAE'], epoch))
            os.makedirs(config.model.saving_path, exist_ok=True)
            save_path = os.path.join(config.model.saving_path, 'model.pth')
            torch.save(model.state_dict(), save_path)

        if early_stopping:
            logger.info('Epoch: {}  Early Stopping'.format(epoch))
            return

        training_controller.epoch_num_plus_1()

        # If you want to test the model at every 5 epochs, set return_metric False.
        if epoch%5==0:
            test(model, test_dataloader, logger, device, return_metric=True)

def main(argv):

    config = FLAGS.config
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # os.makedirs(config.model.saving_path, exist_ok=True)
    os.makedirs(os.path.split(FLAGS.log_path)[0], exist_ok=True)
    
    logger = utils.setup_logger(FLAGS.log_path, 'CVAE', mode='a')

    logger.info("Config Information \n{}".format(config))

    model = make_model(config).to(device)
    
    logger.info("Model Information \n{}".format(model))
    logger.info("Parameter Information: {}".format(sum(p.numel() for p in model.parameters())))

    unified_dataloader = UnifiedDataLoader(config.data.dataset_path, config.data.seq_len, config.data.feature_num, \
                                                    config.model.model_type, config.training.batch_size, \
                                                    config.training.num_workers, config.training.masked_imputation_task)

    train_dataloader, val_dataloader = unified_dataloader.get_train_val_dataloader()
    test_dataloader = unified_dataloader.get_test_dataloader()
    logger.info("Dataloader Done")
    
    if FLAGS.mode == "train":
        # train model
        train(model, train_dataloader, val_dataloader, logger, config, device, test_dataloader)

    # test model for 5 times
    model_path = os.path.join(config.model.saving_path, 'model.pth')
    model.load_state_dict(torch.load(model_path))

    logger.info("load trained model from {}".format(model_path))

    result_dict = {'MAE':[], 'RMSE':[], 'MRE':[], 'MSE':[]}

    test_dict = test(model, test_dataloader, logger, device, return_metric=True)

    for key in result_dict.keys():
        result_dict[key].append(test_dict[key])

    for metric in result_dict.keys():
        logger.info("Test Result [{}]: ({:.4f}±{:.3f})".format(metric, np.mean(result_dict[metric]), np.std(result_dict[metric])))

    return 

if __name__ == '__main__':
    try:
        app.run(main)
    except:
        logger = utils.setup_logger(FLAGS.log_path, 'CVAE', mode='a')
        logger.error(traceback.format_exc()) # error messages to default log file        