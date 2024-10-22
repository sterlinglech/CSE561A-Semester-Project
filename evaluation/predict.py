from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration


class BaseBartModel:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        self.model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

    def predict(self, original_text):
        input_ids = self.tokenizer([original_text], return_tensors="pt", max_length=1024, truncation=True, padding="max_length").input_ids
        outputs = self.model.generate(input_ids=input_ids, max_length=1024, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)