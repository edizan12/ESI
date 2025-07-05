# ESI - GPT-2 Based Turkish Chatbot

This is a simple chatbot project that uses GPT-2 and some example Turkish dialogues to create a language model that can chat in Turkish.

I fine-tuned the GPT-2 model using my own short dataset and built a script that lets you train and chat with the bot.

## What's inside?

- `main.py`: main script for training and chatting
- `train_data.txt`: example question-answer pairs
- `gpt2-finetuned/`: output folder for the fine-tuned model
- `LICENSE`: MIT License
- `.gitignore`: files/folders not included in version control

## How to use

Just run:

```bash
python main.py
If the model doesn't exist, it will train the model using the data in train_data.txt.

If the model is already trained, it will load and start the chat mode.

You can type exit to stop chatting.

Requirements
Make sure you have Python 3.8+ and install these:

bash
Kopyala
Düzenle
pip install transformers torch
Notes
The dataset is very small and just for testing.

You can replace it with a larger Turkish dataset for better results.

This is a personal project and mainly for learning.

License
This project is licensed under the MIT License. You are free to use, modify, and share it.
