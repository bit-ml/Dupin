from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModel, AutoTokenizer
import torch

class TrainableModel:
    def __init__(
        self,
        model_name="bert-base-uncased",
        model_type=AutoModel,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        self.model = model_type.from_pretrained(model_name, **kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

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
        
