import tkinter as tk
from tkinter import messagebox,ttk,scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime

#defining the chatbot GUI
class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        self.load_chatbot_data()
        self.conversation_history = []  # List to store conversation history
        self.waiting_for_order_id = False
        self.waiting_for_order_status = False

    '''setting up the GUI '''
    def setup_gui(self):
        self.master.title("ServiAI — Intelligent Customer Support Chatbot")
        self.master.geometry("500x600")
        self.master.configure(bg="#f0f0f0")

        style=ttk.Style()
        style.theme_use("clam")

        main_frame=tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        #chat history text box
        self.chat_history = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD,width=60,height=25,font=("Arial", 10))
        self.chat_history.pack(padx=10,pady=10,fill=tk.BOTH, expand=True)
        self.chat_history.config(state=tk.DISABLED)  # Make the chat history read-only

        #input and button frames
        input_frame=ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X,pady=5)

        self.user_input = ttk.Entry(input_frame, width=50, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT,padx=(0,5),expand=True,fill=tk.X)
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_button=ttk.Button(
            input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        #button frame for clear,save,help
        button_frame=ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X,pady=5)

        self.clear_button=ttk.Button(
            button_frame, text="Clear Chat", command=self.clear_chat
        )
        self.clear_button.pack(side=tk.LEFT,padx=(0,5))

        self.save_button=ttk.Button(
            button_frame, text="Save Chat", command=self.save_chat
        )
        self.save_button.pack(side=tk.LEFT,padx=(0,5))
        self.save_button.pack(side=tk.LEFT)

        self.help_button=ttk.Button(
            button_frame, text="Help", command=self.show_help
        )
        self.help_button.pack(side=tk.RIGHT)

    #loading chatbot data
    def load_chatbot_data(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents=json.loads(
            open(
                r"intents.json"

            ).read()
        )
        self.words=pickle.load(
            open(
                r"words.pkl","rb"
            )
        )
        self.classes=pickle.load(
            open(
                r"classes.pkl","rb"
            )
        )
        self.model=load_model(
            r"chatbot_model.h5")

    #sending and recieving messages
    def send_message(self):
        user_message=self.user_input.get()
        self.user_input.delete(0, tk.END)  # Clear the input field
        if user_message:
            self.update_chat_history("You: " + user_message)
            bot_response=self.get_bot_response(user_message)
            self.update_chat_history("Bot: " + bot_response)
            self.conversation_history.append({"user": user_message, "bot": bot_response})  # Store the conversation in history

    #updating chat history
    def update_chat_history(self, message):
        self.chat_history.config(state=tk.NORMAL)  # Enable editing
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.see(tk.END)  # Scroll to the end
        self.chat_history.config(state=tk.DISABLED)  # Make the chat history read-only

    #getting bot responses
    def get_bot_response(self,user_message):
        if self.waiting_for_order_id:
            self.waiting_for_order_id = False
            return f"Thank you. I have received your Order ID: {user_message}. I will help you with the wrong product issue."
        if self.waiting_for_order_status:
            self.waiting_for_order_status = False
            return f"Thank you. I have received your Order ID: {user_message}. I will check the order status for you."
        if user_message.lower() in ["exit", "quit", "bye"]:
            return "Goodbye! Have a great day!"
        elif user_message.lower().startswith("search"):
            query = user_message[7:].strip()
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I have opened a web search for '{query}'."
        elif user_message.lower()=="time":
            return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
        else:
            ints=self.predict_class(user_message)
            return self.get_response(ints)
    #cleaning up sentence and bag of words
    def clean_up_sentence(self,sentence):
        return[
            self.lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)
        ]

    def bag_of_words(self,sentence):
        sentence_words=self.clean_up_sentence(sentence)
        bag=[1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    #predicting class and getting response
    def predict_class(self,sentence):
        bow=self.bag_of_words(sentence)
        res=self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD=0.25
        results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x:x[1],reverse=True)
        return [
            {"intent":self.classes[r[0]],"probability":str(r[1])} for r in results
        ]

    def get_response(self,intents_list):
        if not intents_list:
            return "I'm sorry, I didn't understand that. Can you please rephrase?"
        tag=intents_list[0]["intent"]
        if tag == "wrong_item":
            self.waiting_for_order_id = True
            self.waiting_for_order_status = False
        if tag == "order_status":
            self.waiting_for_order_status = True
        for intent in self.intents["intents"]:
            if intent["tag"]==tag:
                return random.choice(intent["responses"])
        return "I'm sorry,I dont have a specific response for that. Can you try asking something else?"

    #clearing and saving chats
    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)  # Enable editing
        self.chat_history.delete(1.0, tk.END)  # Clear the chat history
        self.chat_history.config(state=tk.DISABLED)  # Make the chat history read-only
        self.conversation_history.clear()  # Clear the conversation history

    def save_chat(self):
        filename=(
            f"chat_history_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.txt"
        )

        with open(filename, "w") as f:
            for chat in self.conversation_history:
                f.write(f"You: {chat['user']}\n")
                f.write(f"Bot: {chat['bot']}\n")
                f.write("\n")
        messagebox.showinfo("Chat Saved", f"Chat history saved to {filename}")

    #showing help message
    def show_help(self):
        help_message=(
            "Welcome to ServiAI — Intelligent Customer Support Chatbot!\n\n"
            "You can ask me questions or type commands:\n"
            "- Type 'search <query>' to perform a web search.\n"
            "- Type 'time' to get the current time.\n"
            "- Type 'exit', 'quit', or 'bye' to end the conversation.\n\n"
            "Feel free to ask any other questions, and I'll do my best to assist you!"
        )
        messagebox.showinfo("Help", help_message)

#running the application
if __name__=="__main__":
    root=tk.Tk()
    chatbot_gui=ChatbotGUI(root)
    root.mainloop()