# Chatbot Project

## Overview
This project is a conversational AI chatbot built using Hugging Face's T5 model. The chatbot is fine-tuned on a custom dataset and is capable of generating relevant responses based on user input. It utilizes NLP preprocessing techniques to enhance response quality and coherence.

## Features
- **Fine-tuned T5 model** for chatbot interactions
- **Preprocessing techniques** including tokenization, normalization, and stopword removal
- **Memory-based conversation context** to improve response relevance
- **Predefined responses** for common queries
- **Trained on a custom dataset** using Hugging Face's `Trainer`

## Installation

### Prerequisites
Ensure you have Python installed (recommended version 3.8 or higher). Install the required dependencies:

```bash
pip install torch datasets transformers nltk
```

## Usage

### 1. Train the Chatbot
Run the training script to fine-tune the model on your dataset:

```bash
python chatbot.py
```

### 2. Run the Chatbot
To start a chatbot session, execute:

```bash
python chatbot.py
```

Type your messages and interact with the chatbot! Type `exit` to end the session.

## Project Structure
```
├── chatbot.py                # Main chatbot script
├── chatbot_data.csv          # Training dataset
├── fine_tuned_chatbot/       # Saved fine-tuned model
├── logs/                     # Training logs
├── README.md                 # Project documentation
└── requirements.txt          # Required dependencies
```

## Technical Details

### Model Training
- Uses `google/flan-t5-small` as the base model.
- Custom dataset is loaded from `chatbot_data.csv`.
- Tokenization and preprocessing are handled with `T5Tokenizer`.
- Training is performed using `Trainer` from Hugging Face with the following parameters:
  - `epochs`: 3
  - `batch size`: 8
  - `learning rate`: Default AdamW optimizer settings

### Preprocessing Steps
- Converts text input to lowercase
- Tokenizes input using `nltk`
- Removes stopwords and punctuation
- Normalizes text before training

### Conversation Context
- Maintains the last 5 user-chatbot exchanges to improve response relevance.
- Uses Hugging Face's `pipeline` for text generation.

## Challenges & Solutions
### 1. Response Quality & Coherence
- **Issue**: Some responses lacked relevance.
- **Solution**:
  - Increased fine-tuning epochs.
  - Applied temperature scaling and top-k sampling.
  - Implemented post-processing for chatbot outputs.

### 2. Computational Constraints
- **Issue**: Training required significant resources.
- **Solution**:
  - Used Google Colab for accelerated training.
  - Fine-tuned only essential model layers.

### 3. UI Response Time Optimization
- **Issue**: Delay in response generation.
- **Solution**:
  - Implemented caching mechanisms.
  - Optimized API request handling.
  - Tuned inference parameters.

## Screenshots

![image](https://github.com/user-attachments/assets/16a0173d-102e-4815-8254-c63af3dd7181)

## Next Steps
 Enhance chatbot responses with additional fine-tuning techniques.
 Improve UI with web tools for a better user experience.
 Deploy chatbot on a cloud-based server for wider access.
 Conduct extensive user testing and collect feedback.
 Implement logging and analytics to track chatbot performance.

## Author
Jared Scott

## License
MIT License
