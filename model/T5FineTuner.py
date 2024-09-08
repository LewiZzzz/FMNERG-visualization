import pytorch_lightning as pl
import torch.nn as nn
import torch
import datetime


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, tfm_model, tokenizer):
        super().__init__()
        self.model = tfm_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, vis_feats=None, vis_attention_mask=None, img_label=None,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            vis_feats=vis_feats,
            vis_attention_mask=vis_attention_mask,
            img_label=img_label,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        vis_feats = batch["vis_feats"]
        vis_attention_mask = batch["vis_attention_mask"]
        img_label = batch["img_label"]

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            vis_feats=vis_feats,
            vis_attention_mask=vis_attention_mask,
            img_label=img_label,
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        start = datetime.datetime.now()  # #
        loss = self._step(batch)
        end = datetime.datetime.now()  # #
        diff_time = end - start  # #
        step_time = torch.tensor(diff_time.microseconds, device=loss.device)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs, "step_time": step_time}  # #

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # import pdb;pdb.set_trace()
        sum_step_time = torch.stack([x['step_time'] for x in outputs]).sum() * 1e-6  # # jmwang
        tensorboard_logs = {"avg_train_loss": avg_train_loss, "sum_step_time": sum_step_time}  ##jmwang
        return {"avg_train_loss": avg_train_loss, "sum_step_time": sum_step_time, "log": tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict


