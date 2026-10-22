# ServiAI — Intelligent Customer Support Chatbot

ServiAI is an intelligent customer-support chatbot developed using **Python, Natural Language Processing (NLP), and TensorFlow/Keras**. It classifies user queries into predefined customer-service intents and provides appropriate responses through a desktop GUI.

## 🚀 Features

- NLP-based customer query classification
- Tokenization and lemmatization using NLTK
- Bag-of-Words feature extraction
- Neural-network-based multi-class intent classification
- Tkinter desktop graphical user interface
- Order status and wrong-product conversation handling
- Order ID collection
- Web search command
- Current time command
- Exit/quit/bye commands
- Chat history saving
- Help functionality
- Predefined responses using `intents.json`

## 🧠 How It Works

The chatbot follows this pipeline:

```text
User Input
    ↓
Tkinter GUI
    ↓
Text Preprocessing
    ↓
Tokenization + Lemmatization
    ↓
Bag-of-Words
    ↓
Neural Network
    ↓
Intent Classification
    ↓
Response Selection
    ↓
Bot Response
```

The trained neural network predicts the **intent** of the user's message. The chatbot then selects an appropriate predefined response associated with that intent from `intents.json`.

## 🛠️ Technologies Used

- **Python**
- **NLTK** — Natural Language Processing
- **TensorFlow / Keras** — Neural Network and Model Training
- **NumPy** — Numerical operations
- **Tkinter** — Graphical User Interface
- **JSON** — Intent and response data
- **Pickle** — Storing vocabulary and intent classes

## 🤖 Machine Learning Model

The chatbot uses a feed-forward neural network for multi-class intent classification.

### Model Architecture

```text
Input Layer
    ↓
Dense Layer — 128 neurons + ReLU
    ↓
Dropout — 50%
    ↓
Dense Layer — 64 neurons + ReLU
    ↓
Dropout — 50%
    ↓
Output Layer — Softmax
```

### Training

- Optimizer: SGD with momentum and Nesterov acceleration
- Loss function: Categorical Cross-Entropy
- Epochs: 250
- Batch size: 5
- Output: Intent probability distribution

## 📂 Project Structure

```text
ServiAI-Customer-Support-Chatbot/
│
├── main.py
├── gui.py
├── intents.json
├── chatbot_model.h5
├── words.pkl
├── classes.pkl
├── README.md
└── .gitignore
```

### File Description

| File | Description |
|---|---|
| `main.py` | Preprocesses training data and trains the neural-network model |
| `gui.py` | Runs the Tkinter chatbot interface |
| `intents.json` | Contains intents, example patterns, and predefined responses |
| `chatbot_model.h5` | Trained neural-network model |
| `words.pkl` | Serialized vocabulary |
| `classes.pkl` | Serialized intent classes |
| `.gitignore` | Specifies files that should not be uploaded to Git |

## 💬 Example Interaction

```text
User: Where is my order?

Bot: Please provide your Order ID.

User: 123ABC

Bot: Thank you. I have received your Order ID.
```

The chatbot also supports commands such as:

```text
search Python tutorials
time
bye
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/madhumitha0656/ServiAI-Customer-Support-Chatbot.git
cd ServiAI-Customer-Support-Chatbot
```

### 2. Install the required libraries

```bash
pip install tensorflow nltk numpy keras
```

### 3. Run the chatbot

```bash
python gui.py
```

## ⚠️ Current Limitations

- Uses a Bag-of-Words representation, which does not capture deep semantic meaning or word order.
- The chatbot is limited to predefined customer-support intents.
- Order IDs are currently collected but are not validated against a real database.
- Order status is not connected to a real backend or API.
- The current training pipeline does not use a separate validation/test dataset.
- Responses are predefined rather than generated dynamically.

## 🔮 Future Improvements

- Integrate a real order-management database or API
- Add Order ID validation
- Add train/validation/test evaluation
- Calculate precision, recall and F1-score
- Add a confusion matrix for intent classification
- Improve semantic understanding using transformer-based models
- Add authentication and secure customer information handling
- Deploy the chatbot as a web application or API

## 👩‍💻 Author

**Madhumitha**

GitHub: [madhumitha0656](https://github.com/madhumitha0656)