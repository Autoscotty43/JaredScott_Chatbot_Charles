from chatbot import Chatbot
import time

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot is running! Type 'exit' to quit.")
    print("Charles: Hello! I am Charles, your chatbot. How can I assist you today?") #Added

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Charles: Goodbye!")
            break
        start_time = time.time()
        response = bot.get_response(user_input)
        end_time = time.time()
        print(f"Charles: {response} (Response Time: {end_time - start_time:.2f} seconds)") # timing