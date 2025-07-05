import os
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)

# 1. Örnek sohbet verisini oluştur (train_data.txt)
def create_train_file():
    data = [
        ("Merhaba! Nasılsın?", "Merhaba! Ben iyiyim, teşekkür ederim. Sen nasılsın?"),
        ("Paris'in başkenti neresidir?", "Paris zaten Fransa'nın başkentidir."),
        ("Bugün hava nasıl?", "Bulunduğun konuma göre değişir ama genelde bugün hava güzel görünüyor."),
        ("En iyi film önerin nedir?", "Bu tamamen zevkine bağlı ama 'Inception' gibi bilim kurgu filmleri oldukça beğenilir."),
        ("Python programlama dili hakkında ne biliyorsun?", "Python, öğrenmesi kolay, çok amaçlı bir programlama dilidir ve yapay zeka, web geliştirme gibi birçok alanda kullanılır."),
        ("Kahve mi çay mı daha sağlıklı?", "Her ikisi de ölçülü tüketildiğinde faydalıdır; kahve enerji verir, çay ise antioksidan içerir."),
        ("Neden uyku önemlidir?", "Uyku, vücudun ve beynin dinlenmesi, yenilenmesi için çok önemlidir."),
        ("Bana kısa bir fıkra anlatır mısın?", "Tabii! İki tavşan yolda yürürken biri diğerine: 'Havuç var mı?' diye sormuş, diğeri: 'Hayır, sen neden sordun?' diye cevap vermiş."),
        ("Mars'a gitmek mümkün mü?", "Şu anda Mars'a insan göndermek için çalışmalar devam ediyor ama henüz mümkün değil."),
        ("En popüler sosyal medya platformları hangileri?", "Instagram, Facebook, Twitter ve TikTok şu anda en popüler platformlar arasında."),
    ]
    with open("train_data.txt", "w", encoding="utf-8") as f:
        for prompt, response in data:
            line = f"User: {prompt} Bot: {response}\n"
            f.write(line)

# 2. Dataset oluşturma
def load_dataset(tokenizer, file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
        overwrite_cache=True,
    )

# 3. Model eğitimi
def train():
    # Örnek veri varsa geç, yoksa oluştur
    if not os.path.exists("train_data.txt"):
        create_train_file()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_dataset = load_dataset(tokenizer, "train_data.txt")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")
    print("Model eğitimi tamamlandı ve kaydedildi.")

# 4. Eğitilmiş modelle sohbet etme
def chat():
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    model.eval()

    print("\nChat bot hazır! Çıkmak için 'exit' yazabilirsin.\n")
    while True:
        user_input = input("Sen: ")
        if user_input.lower() == "exit":
            print("Görüşürüz!")
            break
        prompt = f"User: {user_input} Bot:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 100,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(prompt):].strip()
        print(f"Bot: {response}")

# 5. Program ana akışı
if __name__ == "__main__":
    print("GPT-2 Chatbot - Eğitim ve Sohbet")
    if not os.path.exists("./gpt2-finetuned"):
        print("Model bulunamadı. Eğitim başlatılıyor...")
        train()
    else:
        print("Eğitilmiş model bulundu.")
    chat()

