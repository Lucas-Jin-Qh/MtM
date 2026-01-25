import torch
import numpy as np
import wandb
import os
from sklearn.metrics import r2_score as sklearn_r2
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            **kwargs
    ):
        # get all the arguments
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        # get arguments from kwargs if they exist
        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.stitching = kwargs.get("stitching", None)
        self.num_neurons = kwargs.get("num_neurons", None)

        self.model_class = self.config.model.model_class

        if self.config.method.model_kwargs.clf:
            self.metric = 'acc'
        elif self.config.method.model_kwargs.reg:
            self.metric = 'rsquared'
        else:
            self.metric = 'r2'
                
        self.session_active_neurons = []

        self.masking_ratio = model.encoder.masker.ratio
        self.masking_mode = model.encoder.masker.mode
        self.masking_schemes = ['neuron', 'causal']
        if self.masking_mode == "all":
            self.masking_schemes += ['intra-region', 'inter-region']

        if self.masking_mode in ["combined", "all"]:
            print("(train) switch between masking modes: ", self.masking_schemes)

    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        # train loop
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                # if eval_epoch_results[f'eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss}")
                    print(f"epoch: {epoch} best eval trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    # save model
                    self.save_model(name="best", epoch=epoch)
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0], 
                            preds=eval_epoch_results['eval_preds'][0], epoch=epoch,
                            active_neurons=self.session_active_neurons[0][:5]
                        )

                        if self.config.wandb.use:
                            wandb.log({"best_epoch": epoch,
                                    "best_gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                                    "best_r2_fig": wandb.Image(gt_pred_fig['plot_r2'])})

                        else:
                            gt_pred_fig['plot_gt_pred'].savefig(
                                os.path.join(self.log_dir, f"best_gt_pred_fig_{epoch}.png")
                            )
                            gt_pred_fig['plot_r2'].savefig(
                                os.path.join(self.log_dir, f"best_r2_fig_{epoch}.png")
                            )

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

            # save model by epoch
            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                if self.config.method.model_kwargs.method_name == 'ssl':

                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0], 
                        preds=eval_epoch_results['eval_preds'][0], 
                        epoch=epoch,
                        active_neurons=self.session_active_neurons[0][:5]
                    )
                    if self.config.wandb.use:
                        wandb.log({
                            "gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                            "r2_fig": wandb.Image(gt_pred_fig['plot_r2'])
                        })
                    else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"gt_pred_fig_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"r2_fig_{epoch}.png")
                        )

            # wandb log
            if self.config.wandb.use:
                wandb.log({
                    "train_loss": train_epoch_results['train_loss'],
                    "eval_loss": eval_epoch_results['eval_loss'],
                    f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
                })
                
        # save last model
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            wandb.log({"best_eval_loss": best_eval_loss,
                       f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric})
            
    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if self.masking_mode in ["combined", "all"]:
                masking_mode = random.sample(self.masking_schemes, 1)[0]
                if masking_mode == 'temporal':
                    self.model.encoder.masker.ratio = 0.3
                elif masking_mode == 'causal':
                    self.model.encoder.masker.ratio = 0.6
                else:
                    self.model.encoder.masker.ratio = self.masking_ratio
            else:
                masking_mode = self.masking_mode
            outputs = self._forward_model_outputs(batch, masking_mode)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            train_examples += outputs.n_examples
        return{
            "train_loss": train_loss/train_examples
        }
    
    def _forward_model_outputs(self, batch, masking_mode):
        batch = move_batch_to_device(batch, self.accelerator.device)
        return self.model(
            batch['spikes_data'], 
            time_attn_mask=batch['time_attn_mask'],
            space_attn_mask=batch['space_attn_mask'],
            spikes_timestamps=batch['spikes_timestamps'], 
            spikes_spacestamps=batch['spikes_spacestamps'], 
            targets = batch['target'],
            neuron_regions=batch['neuron_regions'],
            masking_mode=masking_mode, 
            spike_augmentation=self.config.data.spike_augmentation,
            num_neuron=batch['spikes_data'].shape[2],
            eid=batch['eid'][0]  # each batch consists of data from the same eid
        ) 
    
    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0.
        eval_examples = 0
        session_results = {}
        for num_neuron in self.num_neurons:
            session_results[num_neuron] = {
                "gt": [],
                "preds": []
            }
        if self.eval_dataloader:
            gt, preds = [], []
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    if self.masking_mode in ["combined", "all"]:
                        masking_mode = random.sample(self.masking_schemes, 1)[0]
                        if masking_mode == 'temporal':
                            self.model.encoder.masker.ratio = 0.3
                        elif masking_mode == 'causal':
                            self.model.encoder.masker.ratio = 0.6
                        else:
                            self.model.encoder.masker.ratio = self.masking_ratio
                    else:
                        masking_mode = self.masking_mode
                    outputs = self._forward_model_outputs(batch, masking_mode)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_examples += outputs.n_examples
                    if self.model_class in ['NDT1', 'iTransformer']:
                        num_neuron = batch['spikes_data'].shape[2]
                    elif self.model_class in ['NDT2', 'STPatch']:
                        num_neuron = outputs.num_neuron
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        session_results[num_neuron]["gt"].append(outputs.targets.clone()[:,:,:num_neuron])
                        session_results[num_neuron]["preds"].append(outputs.preds.clone()[:,:,:num_neuron])
                    else:
                        session_results[num_neuron]["gt"].append(outputs.targets.clone())
                        session_results[num_neuron]["preds"].append(outputs.preds.clone())
                    
            results_list = []
            for idx, num_neuron in enumerate(self.num_neurons):
                _gt = torch.cat(session_results[num_neuron]["gt"], dim=0)
                _preds = torch.cat(session_results[num_neuron]["preds"], dim=0)

                if self.config.method.model_kwargs.loss == "poisson_nll":
                    # model outputs are log-rate (log lambda). Convert to rate via exp.
                    # keep a copy of raw predictions (log-rate) for diagnostics
                    raw_preds = _preds.clone()
                    pred_rates = torch.exp(_preds)
                    # If binsize is available in config, convert rates (Hz) -> expected counts per bin
                    binsize = None
                    try:
                        binsize = getattr(self.config.data, "binsize", None)
                    except Exception:
                        binsize = None
                    if binsize is not None:
                        _preds = pred_rates * float(binsize)
                        # _preds now in expected counts per bin, matching gt which is binned counts
                    else:
                        # Fall back to using rates if binsize unknown
                        _preds = pred_rates
                    # one-time diagnostic dump: compare raw vs exp vs scaled R2 and save small samples
                    try:
                        if (not hasattr(self, "_r2_diagnostic_done")) and (self.log_dir is not None):
                            os.makedirs(self.log_dir, exist_ok=True)
                            # take up to 100 trials for diagnostics
                            n_sample = min(100, _gt.shape[0])
                            gt_sample = _gt[:n_sample].detach().cpu().numpy()
                            raw_sample = raw_preds[:n_sample].detach().cpu().numpy()
                            exp_sample = np.exp(raw_sample)
                            if binsize is not None:
                                scaled_sample = exp_sample * float(binsize)
                            else:
                                scaled_sample = exp_sample
                            # flatten for r2 computation
                            gt_flat = gt_sample.reshape(-1)
                            raw_flat = raw_sample.reshape(-1)
                            exp_flat = exp_sample.reshape(-1)
                            scaled_flat = scaled_sample.reshape(-1)
                            # baseline mean predictor
                            baseline_flat = np.full_like(gt_flat, np.nanmean(gt_flat))
                            # compute r2 with sklearn (gt, pred)
                            try:
                                r2_raw = sklearn_r2(gt_flat, raw_flat)
                            except Exception:
                                r2_raw = np.nan
                            try:
                                r2_exp = sklearn_r2(gt_flat, exp_flat)
                            except Exception:
                                r2_exp = np.nan
                            try:
                                r2_scaled = sklearn_r2(gt_flat, scaled_flat)
                            except Exception:
                                r2_scaled = np.nan
                            try:
                                r2_base = sklearn_r2(gt_flat, baseline_flat)
                            except Exception:
                                r2_base = np.nan
                            print("DIAG R2 (gt vs raw_pred/log):", r2_raw)
                            print("DIAG R2 (gt vs exp(raw_pred)):", r2_exp)
                            print("DIAG R2 (gt vs scaled exp):", r2_scaled)
                            print("DIAG R2 (baseline mean):", r2_base)
                            # save arrays
                            np.save(os.path.join(self.log_dir, "diag_gt_sample.npy"), gt_sample)
                            np.save(os.path.join(self.log_dir, "diag_raw_pred_sample.npy"), raw_sample)
                            np.save(os.path.join(self.log_dir, "diag_exp_pred_sample.npy"), exp_sample)
                            np.save(os.path.join(self.log_dir, "diag_scaled_pred_sample.npy"), scaled_sample)
                            np.save(os.path.join(self.log_dir, "diag_r2_values.npy"), np.array([r2_raw, r2_exp, r2_scaled, r2_base], dtype=np.float64))
                            self._r2_diagnostic_done = True
                    except Exception:
                        pass
                elif self.config.method.model_kwargs.loss == "cross_entropy" :
                    _preds = torch.nn.functional.softmax(_preds, dim=1)
                gt.append(_gt)
                preds.append(_preds)

                if len(self.session_active_neurons) < len(self.num_neurons):
                    active_neurons = np.argsort(gt[idx].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    self.session_active_neurons.append(active_neurons)
                if self.config.method.model_kwargs.method_name == 'ssl':
                    results = metrics_list(gt = gt[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0),
                                        pred = preds[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0), 
                                        metrics=["r2"], 
                                        device=self.accelerator.device)
                    
                elif self.config.method.model_kwargs.method_name == 'sl':
                    if self.config.method.model_kwargs.clf:
                        results = metrics_list(gt = gt[idx].argmax(1),
                                            pred = preds[idx].argmax(1), 
                                            metrics=[self.metric], 
                                            device=self.accelerator.device)
                    elif self.config.method.model_kwargs.reg:
                        results = metrics_list(gt = gt[idx],
                                            pred = preds[idx],
                                            metrics=[self.metric],
                                            device=self.accelerator.device)
                results_list.append(results[self.metric])

        return {
            "eval_loss": eval_loss/eval_examples,
            f"eval_trial_avg_{self.metric}": np.mean(results_list),
            "eval_gt": gt,
            "eval_preds": preds,
        }
    
    def plot_epoch(self, gt, preds, epoch, active_neurons):
        # Prepare arrays used for plotting (trial-averaged)
        gt_mean = gt.mean(0)             # (T, N) or (T, ...) depending on shape
        preds_mean = preds.mean(0)

        gt_arr = gt_mean.T.cpu().numpy()
        pred_arr = preds_mean.T.detach().cpu().numpy()

        # Debug printing removed

        # If binsize is available in config, print aligned stats for quick unit check
        binsize = None
        try:
            if hasattr(self, "config") and hasattr(self.config, "data"):
                binsize = getattr(self.config.data, "binsize", None)
        except Exception:
            binsize = None

        if binsize is not None:
            try:
                print("DEBUG binsize:", binsize)
                # convert GT counts -> rate and pred (log-rate) -> expected counts in bin
                gt_rate_mean = (gt_arr / binsize).mean()
                pred_counts_mean = (np.exp(pred_arr) * binsize).mean()
                print("DEBUG after aligning units (means): gt_rate_mean, pred_counts_mean:", float(gt_rate_mean), float(pred_counts_mean))
            except Exception:
                pass
        # Detailed R2 alignment checks: compare counts vs pred_counts and rate vs pred_rate
        try:
            def mean_r2(y_true, y_pred):
                # y_true, y_pred: (N, T)
                rs = []
                for i in range(y_true.shape[0]):
                    yt = y_true[i]
                    yp = y_pred[i]
                    denom = np.sum((yt - yt.mean()) ** 2)
                    if denom == 0:
                        continue
                    num = np.sum((yt - yp) ** 2)
                    rs.append(1.0 - (num / denom))
                return np.nan if len(rs) == 0 else float(np.nanmean(rs))

            gt_counts = gt_arr
            pred_log = pred_arr
            pred_rate = np.exp(pred_log)
            if binsize is None:
                binsize = getattr(self.config.data, "binsize", None) if hasattr(self, "config") else None
            if binsize is None:
                # can't align to counts; still compute rate-vs-rate R2 assuming gt is already rate
                r2_rate = mean_r2(gt_counts, pred_rate)
                print("DEBUG R2 (rate vs pred_rate) (assume GT is rate):", r2_rate)
            else:
                gt_rate = gt_counts / binsize
                pred_counts = pred_rate * binsize
                r2_counts = mean_r2(gt_counts, pred_counts)
                r2_rate = mean_r2(gt_rate, pred_rate)
                print("DEBUG R2 (counts vs pred_counts):", r2_counts)
                print("DEBUG R2 (rate vs pred_rate):", r2_rate)

            # More diagnostics on pred distribution
            try:
                percentiles = np.percentile(pred_log.flatten(), [50, 90, 99, 99.9, 100])
                print("DEBUG pred_log percentiles (50/90/99/99.9/100):", percentiles)
                large_count = int(np.sum(pred_rate > 1000))
                print("DEBUG num bins with pred_rate > 1000:", large_count)
            except Exception:
                pass
        except Exception:
            pass

        gt_pred_fig = plot_gt_pred(gt=gt_arr, pred=pred_arr, epoch=epoch)

        r2_fig = plot_neurons_r2(
            gt=gt_mean,
            pred=preds_mean,
            neuron_idx=active_neurons,
            epoch=epoch,
        )

        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig,
        }
        

    def save_model(self, name="last", epoch=0):
        # save model
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))
        
