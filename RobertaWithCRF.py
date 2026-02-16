import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel
from torchcrf import CRF


class RobertaWithCRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 1. Load the base RoBERTa model
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 2. The Classifier (Emissions)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 3. The CRF Layer
        # batch_first=True means it expects inputs as (batch, seq_len, tags)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            **kwargs
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Mask must be uint8 (byte) for pytorch-crf
            mask = attention_mask.type(torch.uint8) if attention_mask is not None else None

            # === CRITICAL FIX START ===
            # Hugging Face uses -100 for ignored regions (padding/special tokens).
            # pytorch-crf crashes on -100. We must replace -100 with a valid ID (e.g., 0).
            # The 'mask' we created above ensures these positions are NOT trained on,
            # so changing the label value here is safe and necessary to prevent the crash.
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0

            # Pass clean_labels instead of labels
            log_likelihood = self.crf(emissions, clean_labels, mask=mask, reduction='mean')
            # === CRITICAL FIX END ===

            loss = -log_likelihood

        return (loss, emissions) if loss is not None else (emissions,)

    def decode(self, input_ids, attention_mask=None):
        """
        Custom prediction method to use Viterbi decoding.
        """
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(outputs[0])
        mask = attention_mask.type(torch.uint8) if attention_mask is not None else None

        # This returns the best tag SEQUENCE (List of Lists), not just numbers
        best_tags_list = self.crf.decode(emissions, mask=mask)
        return best_tags_list