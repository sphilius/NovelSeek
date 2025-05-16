import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn import to_hetero
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
import numpy as np
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
import torch.distributed as dist
from src.utils import seed_worker

from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.cuda.amp import autocast
from .utils import count_parameters, AverageMeter, AVGMeter, Reporter, Timer

# torch.autograd.set_detect_anomaly(True)

class Oven(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.ngpus = cfg.get('ngpus', 1)



    def _init_training_wt_checkpoint(self, filepath_ckp):
        if not os.path.exists(filepath_ckp):
            return np.Infinity, -1, 0
        
        checkpoint_resum = torch.load(filepath_ckp)
        self.model.load_state_dict(checkpoint_resum['model_state'])
        epoch = checkpoint_resum['epoch']
        previous_best = checkpoint_resum['best_performance']
        previous_best_epoch = checkpoint_resum["best_epoch"]
        previous_best_metrics = checkpoint_resum["local_best_metrics"]
        return previous_best, previous_best_epoch, epoch, previous_best_metrics

    def _init_optim(self):
        if self.cfg['train'].get("optimizer_type", "Adam").lower() in "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=float(self.cfg['train']['learning_rate']),
                                   weight_decay=self.cfg['train'].get("weight_decay", 1e-5)
                                   )
        else: # SGD by defalut
            optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.cfg['train']['learning_rate'], 
                                momentum=self.cfg['train'].get("momentum", 0.9), 
                                weight_decay=self.cfg['train'].get("weight_decay", 1e-5))

        # scheduler = StepLR(optimizer, step_size=int(self.cfg['train']['epochs']*2/3), gamma=0.1)
        if self.cfg['scheduler']['type'] == 'Cosine':
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=self.cfg['train']['epochs'],
                                          eta_min=float(self.cfg['scheduler']['eta_min']))
        elif self.cfg['scheduler']['type'] == 'Exponential':
            scheduler = ExponentialLR(optimizer, gamma=self.cfg['scheduler']['gamma'], last_epoch=-1, verbose=False)
        elif self.cfg['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5)
        else: # otherwise: Fixed lr
            scheduler = None
        return optimizer, scheduler

    def _init_data(self):
        train_dataset = self.get_dataset(**self.cfg['data']['train'])
        val_dataset = self.get_dataset(**self.cfg['data']['val'])

        if not self.cfg['distributed']:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg['data']['batch_size'],
                num_workers=self.cfg['data']['num_workers'],
                shuffle=True,
                worker_init_fn=seed_worker,
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg['data'].get("batch_size_test", self.cfg['data']['batch_size']),
                num_workers=self.cfg['data']['num_workers'],
                shuffle=False,
                drop_last=True,
                worker_init_fn=seed_worker
            )
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(train_dataset, 
                                  batch_size=self.cfg['data']['batch_size'], 
                                  num_workers=self.cfg['data']['num_workers'], 
                                  sampler=train_sampler,
                                  drop_last=True,
                                  worker_init_fn=seed_worker)
            
            valid_sampler = DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(val_dataset, 
                                      batch_size=self.cfg['data'].get("batch_size_test", self.cfg['data']['batch_size']), 
                                      num_workers=self.cfg['data']['num_workers'], 
                                      sampler=valid_sampler, 
                                      drop_last=True,
                                      worker_init_fn=seed_worker)

        return train_loader, val_loader

    def get_dataset(self, dataset_type, **kwargs):
        if dataset_type == 'PowerFlowDataset':
            from src.dataset.powerflow_dataset import PowerFlowDataset
            return PowerFlowDataset(
                data_root=kwargs['data_root'],
                split_txt=kwargs['split_txt'],
                pq_len=kwargs['pq_len'],
                pv_len=kwargs['pv_len'],
                slack_len=kwargs['slack_len'],
                mask_num=kwargs['mask_num']
            )


    def summary_epoch(self,
                      epoch,
                      train_loss, train_matrix,
                      valid_loss, val_matrix,
                      timer, local_best, 
                      local_best_ep=-1,
                      local_best_metrics={},
                      local_best_ema=100, 
                      local_best_ep_ema=-1,
                      local_best_metrics_ema = {},
                      valid_loss_ema=None, val_matrix_ema=None):

        if self.cfg['distributed']:
            if dist.get_rank() == 0:
                cur_lr = self.optim.param_groups[0]["lr"]
                # self.reporter.record({'epoch': epoch+1, 'train_loss': train_loss, 'valid_loss': valid_loss, 'lr': cur_lr})
                self.reporter.record({'loss/train_loss': train_loss}, epoch=epoch)
                self.reporter.record({'loss/val_loss': valid_loss}, epoch=epoch)
                self.reporter.record({'lr': cur_lr}, epoch=epoch)
                self.reporter.record(train_matrix, epoch=epoch)
                self.reporter.record(val_matrix, epoch=epoch)

                # logger.info(f"Epoch {str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']}, lr: {cur_lr: .8f}, eta: {timer.eta}h, train_loss: {train_loss: .5f}, valid_loss: {valid_loss: .5f}")
                logger.info(f"Epoch {str(epoch+1).zfill(3)}/{self.cfg['train']['epochs']},"
                        + f" lr: {cur_lr: .8f}, eta: {timer.eta}h, "
                        + f"train_loss: {train_loss.agg(): .5f}, "
                        + f"valid_loss: {valid_loss.agg(): .5f}")
                
                train_matrix_info = "Train: "
                for key in train_matrix.keys():
                    tkey = str(key).split("/")[-1]
                    train_matrix_info += f"{tkey}:{train_matrix[key].agg(): .6f}  "
                logger.info(f"\t{train_matrix_info}")

                val_matrix_info = "ZTest: "
                performance_record = dict()
                for key in val_matrix.keys():
                    tkey = str(key).split("/")[-1]
                    val_matrix_info += f"{tkey}:{val_matrix[key].agg(): .6f}  "
                    performance_record[key] = val_matrix[key].agg()
                logger.info(f"\t{val_matrix_info}")

                if val_matrix_ema is not None:
                    val_matrix_info_ema = "ZTest-ema: "
                    performance_record_ema = dict()
                    for key in val_matrix_ema.keys():
                        tkey = str(key).split("/")[-1]
                        val_matrix_info_ema += f"{tkey}:{val_matrix_ema[key].agg(): .6f}  "
                        performance_record_ema[key] = val_matrix_ema[key].agg()
                    logger.info(f"\t{val_matrix_info_ema}")

                    checked_performance_ema = {x:y for x,y in performance_record_ema.items() if "rmse" in x}
                    best_performance_ema = max(checked_performance_ema.values())
                    if best_performance_ema < local_best_ema:
                        local_best_ema = best_performance_ema
                        local_best_ep_ema = epoch
                        local_best_metrics_ema = checked_performance_ema
                    logger.info(f"\t           ValOfEMA:{best_performance_ema:.6f}/{local_best_ema:.6f},  Epoch:{epoch+1}/{local_best_ep_ema+1}")
                
                # best_performance = max(performance_record.values())
                checked_performance = {x:y for x,y in performance_record.items() if "rmse" in x}
                best_performance = max(checked_performance.values())
                if best_performance < local_best:
                    local_best = best_performance
                    local_best_metrics = checked_performance
                    local_best_ep = epoch
                    # torch.save(self.model.module, os.path.join(self.cfg['log_path'], 'ckpt_{}_{}.pt'.format(epoch, round(local_best,4))))
                    torch.save(self.model.module, os.path.join(self.cfg['log_path'], 'ckpt_best.pt'))
                
                state = {
                    "epoch": epoch + 1,
                    # "model_state": self.model.module.state_dict(),
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optim.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "best_performance": local_best,
                    "best_epoch":local_best_ep,
                    "local_best_metrics": local_best_metrics,
                }
                torch.save(state, os.path.join(self.cfg['log_path'], 'ckpt_latest.pt'))
                logger.info(f"\tTime(ep):{int(timer.elapsed_time)}s,  Val(curr/best):{best_performance:.6f}/{local_best:.6f},  Epoch(curr/best):{epoch+1}/{local_best_ep+1}")
            # else:
            #     return local_best, local_best_ep
        else:
            cur_lr = self.optim.param_groups[0]["lr"]
            self.reporter.record({'loss/train_loss': train_loss}, epoch=epoch)
            self.reporter.record({'loss/val_loss': valid_loss}, epoch=epoch)
            self.reporter.record({'lr': cur_lr}, epoch=epoch)
            self.reporter.record(train_matrix, epoch=epoch)
            self.reporter.record(val_matrix, epoch=epoch)

            logger.info(f"Epoch {epoch}/{self.cfg['train']['epochs']},"
                        + f" lr: {cur_lr: .8f}, eta: {timer.eta}h, "
                        + f"train_loss: {train_loss.agg(): .5f}, "
                        + f"valid_loss: {valid_loss.agg(): .5f}")

            train_matrix_info = "Train: "
            for key in train_matrix.keys():
                tkey = str(key).split("/")[-1]
                train_matrix_info += f"{tkey}:{train_matrix[key].agg(): .8f}  "
            logger.info(f"\t{train_matrix_info}")

            val_matrix_info = "ZTest: "
            performance_record = dict()
            for key in val_matrix.keys():
                tkey = str(key).split("/")[-1]
                val_matrix_info += f"{tkey}:{val_matrix[key].agg(): .8f}  "
                performance_record[key] = val_matrix[key].agg()
            logger.info(f"\t{val_matrix_info}")

            if val_matrix_ema is not None:
                val_matrix_info_ema = "ZTest-ema: "
                performance_record_ema = dict()
                for key in val_matrix_ema.keys():
                    tkey = str(key).split("/")[-1]
                    val_matrix_info_ema += f"{tkey}:{val_matrix_ema[key].agg(): .6f}  "
                    performance_record_ema[key] = val_matrix_ema[key].agg()
                logger.info(f"\t{val_matrix_info_ema}")
                
                checked_performance_ema = {x:y for x,y in performance_record_ema.items() if "rmse" in x}
                best_performance_ema = max(checked_performance_ema.values())
                if best_performance_ema < local_best_ema:
                    local_best_ema = best_performance_ema
                    local_best_metrics_ema = checked_performance_ema
                    local_best_ep_ema = epoch
                logger.info(f"\t           ValOfEMA:{best_performance_ema:.6f}/{local_best_ema:.6f},  Epoch:{epoch+1}/{local_best_ep_ema+1}")

            # best_performance = max(performance_record)
            checked_performance = {x:y for x,y in performance_record.items() if "rmse" in x}
            best_performance = max(checked_performance.values())
            if best_performance < local_best:  # save best
                local_best = best_performance
                local_best_ep = epoch
                local_best_metrics = checked_performance
                # torch.save(self.model, os.path.join(self.cfg['log_path'], 'ckpt_{}_{}.pt'.format(epoch, round(local_best,4))))
                torch.save(self.model, os.path.join(self.cfg['log_path'], 'ckpt_best.pt'))
            state = {
                "epoch": epoch + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optim.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_performance": local_best,
                "best_epoch":local_best_ep,
                "local_best_metrics": local_best_metrics, 
            }
            torch.save(state, os.path.join(self.cfg['log_path'], 'ckpt_latest.pt'))
            logger.info(f"\tTime(ep):{int(timer.elapsed_time)}s,  Val(curr/best):{best_performance:.6f}/{local_best:.6f},  Epoch(curr/best):{epoch+1}/{local_best_ep+1}")
        
        if val_matrix_ema is not None:
            return local_best, local_best_ep, local_best_ema, local_best_ep_ema, local_best_metrics_ema 
        else:
            return local_best, local_best_ep, local_best_metrics

