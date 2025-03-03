from chatbot import Chatbot

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot is running! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = bot.get_response(user_input)
        print(f"Chatbot: {response}")