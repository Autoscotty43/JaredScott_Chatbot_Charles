from transformers import pipeline

class Chatbot:
    def __init__(self):
        print("Loading AI model...")
        self.model = pipeline("text2text-generation", model="google/flan-t5-small")
        self.conversation_history = []  # Memory

    def get_response(self, user_input):
        self.conversation_history.append(f"You: {user_input}")
        context = " ".join(self.conversation_history[-5:])  # Keep last 5 interactions
        response = self.model(context, max_length=50, truncation=True)
        bot_response = response[0]['generated_text']
        
        self.conversation_history.append(f"Chatbot: {bot_response}")
        return bot_response

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = bot.get_response(user_input)
        print(f"Chatbot: {response}")