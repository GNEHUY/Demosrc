

#  model name : intfloat/e5-mistral-7b-instruct

class Model:
    def __init__(self, config, encoder):
        self.config = config
        self.encoder = encoder
    
    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)