from transformers import AutoModelForMaskedLM, AutoTokenizer

class TrainablePromptModel:
    def __init__(
        self, model_name="bert-base-uncased", model_type=AutoModelForMaskedLM, **kwargs
    ):
        self.model = model_type.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
