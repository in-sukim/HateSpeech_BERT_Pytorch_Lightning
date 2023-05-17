from model import *

class hate_classifier():
    def __init__(self):
        self.config = json.load(open('./hate_model_config.json'))
        self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        self.model = HateClassifier(self.config)
        self.model.load_state_dict(torch.load(f = './hate_clf.pth'))
    
    def predict(self, text):
        output = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length= 80,
            return_attention_mask = True,
            return_token_type_ids=False
            )
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        loss, output = self.model(input_ids, attention_mask)

        prediction = torch.argmax(output, dim=1).numpy().tolist()
        if prediction[0] == 0:
            prediction = '정상댓글'
        else:
            prediction = '혐오댓글'

        return prediction