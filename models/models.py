from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
)
import torch
from torch.nn.functional import softmax


class TrainableModel:
    def __init__(
        self,
        model_name="bert-base-uncased",
        model_type=AutoModel,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        self.model_name = model_name
        self.model = model_type.from_pretrained(model_name, **kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.device = device

    def __call__(self, input_data):
        return self.model(**input_data)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)


class TrainablePromptModel(TrainableModel):
    def __init__(
        self,
        model_name="bert-base-uncased",
        model_type=AutoModelForMaskedLM,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        super().__init__(model_name, model_type, device, **kwargs)


class TrainableClfModel(TrainableModel):
    def __init__(
        self,
        model_name="bert-base-uncased",
        model_type=AutoModelForSequenceClassification,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        super().__init__(model_name, model_type, device, **kwargs)

    def predict(self, input_data):
        batch_iid = input_data[0]
        batch_am = input_data[1]
        batch_tti = input_data[2]

        return softmax(
            self.model(
                {
                    "input_ids": batch_iid.to(self.device),
                    "attention_mask": batch_am.to(self.device),
                    "token_type_ids": batch_tti.to(self.device),
                }
            )[0]
        )
