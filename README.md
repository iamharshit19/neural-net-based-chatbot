## Streamlit Smart Chatbot

This project is an intelligent, conversational chatbot with a web-based user interface built using Streamlit. The chatbot uses a neural network trained with TensorFlow/Keras to understand user intent and can handle both general conversation and mathematical calculations.

✨ Features
Conversational AI: Understands natural language to respond to greetings, questions, and goodbyes.

Secure Expression Evaluation: Uses the asteval library to safely compute math problems, preventing security risks from malicious code.

Interactive Web UI: A clean, user-friendly chat interface powered by Streamlit.

Extensible Knowledge Base: Easily add new conversational abilities by modifying a simple JSON file and retraining the model.

🔧 How It Works
The project is broken down into three main components:

intents.json (The Brains)
This JSON file is the chatbot's knowledge base. It contains predefined categories (or "intents") with patterns of user input and corresponding responses. For the math intent, it includes templates to format the calculated result.

training.py (The Teacher)
This script is responsible for training the neural network. It reads the intents.json file, processes the text using NLTK (tokenizing and lemmatizing), and then builds and trains a sequential neural network. The script saves the trained model (chatbot_model.h5) and the vocabulary (words.pkl, classes.pkl).

streamlit_app.py (The Face)
This is the main application that users interact with. It loads the trained model and vocabulary, provides the Streamlit chat interface, and contains the core logic to:

Predict the user's intent.

If the intent is "mathematics," it extracts and solves the expression.

If the intent is conversational, it selects an appropriate response.

Display the conversation in the web app.

🚀 Getting Started
Follow these instructions to set up and run the chatbot on your local machine.

Prerequisites
Python 3.7 or higher

pip package manager

1. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to keep the project's dependencies isolated.

# Create the environment
      python -m venv venv

# Activate on Mac/Linux
      source venv/bin/activate

# Activate on Windows
      .\venv\Scripts\activate

2. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

            pip install -r requirements.txt

3. Train the Chatbot Model
Before you can run the app, you must train the model. This script reads your intents.json file and creates the necessary model files.

            python training.py

This will create chatbot_model.h5, words.pkl, and classes.pkl. You only need to re-run this script when you modify intents.json.

4. Run the Streamlit App
Now you are ready to start the chatbot!

            streamlit run streamlit_app.py

This will launch the application in your default web browser. You can now start chatting with your bot.

💬 Usage
Once the app is running, you can interact with the chatbot in two main ways:

General Conversation:

Hello

What can you do?

Thanks


🛠️ How to Customize
You can easily teach the chatbot new things.

Open intents.json: Add a new intent block or add more patterns to an existing one.

      {
      "tags": "your_new_intent",
      "patterns": [
    "A new phrase you want the bot to learn",
    "Another example of the phrase"
    ],
    "responses": [
      "The bot's new reply"
    ]
    }

Retrain the Model: After saving your changes to intents.json, you must run the training script again.

    python training.py

Restart the App: If your Streamlit app is still running, stop it (Ctrl+C in the terminal) and restart it to see your changes.

    streamlit run streamlit_app.py

<img width="1920" height="1080" alt="chatbot" src="https://github.com/user-attachments/assets/9e72a388-06ba-4e2e-a875-213f74116743" />

