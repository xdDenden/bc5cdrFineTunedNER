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

    def _apply_transition_constraints(self):
        """
        Forcefully penalize illegal BIO transitions so Viterbi avoids them.
        Tags: "O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4
        """
        with torch.no_grad():
            # Prevent O -> Inside
            self.crf.transitions[0, 3] = -10000.0  # O -> I-Disease
            self.crf.transitions[0, 4] = -10000.0  # O -> I-Chemical

            # Prevent Cross-Entity Jumps
            self.crf.transitions[1, 3] = -10000.0  # B-Chemical -> I-Disease
            self.crf.transitions[2, 4] = -10000.0  # B-Disease -> I-Chemical
            self.crf.transitions[4, 3] = -10000.0  # I-Chemical -> I-Disease
            self.crf.transitions[3, 4] = -10000.0  # I-Disease -> I-Chemical

            # Prevent sequences from starting with an 'Inside' tag
            self.crf.start_transitions[3] = -10000.0 # Start -> I-Disease
            self.crf.start_transitions[4] = -10000.0 # Start -> I-Chemical

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

        # Apply constraints before calculating the loss
        self._apply_transition_constraints()

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

        # Apply constraints before decoding
        self._apply_transition_constraints()

        best_tags_list = self.crf.decode(emissions, mask=mask)
        return best_tags_list