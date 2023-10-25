import lightning as L
import torch
import torch.optim as optim
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)


class TwitchModel(L.LightningModule):
    def __init__(
        self,
        model_name,
        num_training_steps,
        learning_rate,
        num_warmup_steps,
        eps,
        weight_decay
    ):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.eps = eps
        self.weight_decay = weight_decay

    def forward(self, batch_encodings):
        outputs = self.classifier(**batch_encodings)
        return outputs

    def training_step(self, batch, batch_idx=0):
        twitch_items, labels, batch_encodings = batch.values()
        outputs = self.forward(batch_encodings)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels)

        predictions = torch.argmax(logits.detach(), dim=1).tolist()
        for item, prediction in zip(twitch_items, predictions):
            item.prediction = prediction

        return {'twitch_items': twitch_items, 'loss': loss}

    def validation_step(self, batch, batch_idx=0):
        twitch_items, labels, batch_encodings = batch.values()
        outputs = self.forward(batch_encodings)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels)

        predictions = torch.argmax(logits.detach(), dim=1).tolist()
        for item, prediction in zip(twitch_items, predictions):
            item.prediction = prediction

        return {'twitch_items': twitch_items, 'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            eps=self.eps,
            weight_decay=self.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
