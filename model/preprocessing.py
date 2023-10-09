import lightning as L
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


class PreprocessingModel(L.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            output_hidden_states=True
        )

    def forward(self, batch_encodings):
        outputs = self.classifier(**batch_encodings)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.forward(batch['batch_encodings'])
        embeddings = outputs.hidden_states[-1][:, 0]
        classifications = torch.softmax(outputs.logits, dim=-1)
        return {'embeddings': embeddings, 'classifications': classifications}


def batch_cosine_similarity(batch_good_embeddings, batch_bad_embeddings):
    batch_good_embeddings = batch_good_embeddings.unsqueeze(1)
    batch_bad_embeddings = batch_bad_embeddings.unsqueeze(0)
    batch_cosine_similarities = F.cosine_similarity(batch_good_embeddings, batch_bad_embeddings, dim=-1)
    return batch_cosine_similarities


def similar_message_indices(good_embeddings, bad_embeddings):
    cosine_similarities = [
        [
            batch_cosine_similarity(batch_bad_embedding, batch_good_embedding)
            for batch_good_embedding in good_embeddings
        ]
        for batch_bad_embedding in bad_embeddings
    ]

    good_message_indices = [
        torch.argmax(torch.cat(bad_embedding, dim=-1), dim=-1)
        for bad_embedding in cosine_similarities
    ]

    good_message_indices = torch.cat(good_message_indices).tolist()
    return good_message_indices


def false_positive_message_indices(classifications):
    classifications = torch.cat(classifications)
    good_messages_indices = [
        index for index, probabilty in enumerate(classifications)
        if probabilty[1].item() > 0.95
    ]

    return good_messages_indices
