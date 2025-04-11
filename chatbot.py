from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from responses import predefined_responses
from nlp_utils import clean_response
import torch

class Chatbot:
    def __init__(self, model_name="google/flan-t5-small"):
        print(f"Loading pre-trained AI model: {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available(): #use GPU if available
            self.model = self.model.to("cuda")
        self.conversation_history = []
        self.max_history_len = 5 #added

    def get_response(self, user_input):
        user_input_lower = user_input.lower()

        if user_input_lower in predefined_responses:
            return predefined_responses[user_input_lower]

        processed_input = f"Charles: {user_input}"
        input_ids = self.tokenizer.encode(processed_input, return_tensors="pt", max_length=128, truncation=True)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        output = self.model.generate(input_ids,
                                     max_length=50,
                                     num_beams=5,
                                     no_repeat_ngram_size=2,
                                     top_k=50,
                                     top_p=0.95,
                                     do_sample=True)
        bot_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        bot_response = clean_response(bot_response)
        self.conversation_history.append(f"You: {user_input}")
        self.conversation_history.append(f"Charles: {bot_response}")
        # Keep only the last few turns to manage context window.
        self.conversation_history = self.conversation_history[-self.max_history_len:]
        return bot_response