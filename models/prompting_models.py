from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class TrainablePromptModel:
    def __init__(
        self,
        model_name="bert-base-uncased",
        model_type=AutoModelForMaskedLM,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        self.model = model_type.from_pretrained(model_name, **kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    def __call__(self, input):
        return self.model(**input)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)