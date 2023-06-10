import requests
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# API endpoint
url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

# Request parameters
params = {}

try:
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        cards = data["data"]

        card_names = [card["name"] for card in cards]
        card_descriptions = [card["desc"] for card in cards]

        # Fine-tuning GPT model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        # Prepare card name data for fine-tuning
        card_data = card_names

        # Tokenize and convert the card data to input tensors
        input_ids = tokenizer.batch_encode_plus(
            card_data,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

        # Fine-tuning parameters
        epochs = 3
        learning_rate = 1e-5

        # Set device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model and input tensors to the device
        model.to(device)
        input_ids = input_ids.to(device)

        # Fine-tuning loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # Save the fine-tuned model
        model.save_pretrained("fine_tuned_model")
        tokenizer.save_pretrained("fine_tuned_model")

        # Generate new card names
        model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")
        tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model")

        # Set the generation parameters
        num_sequences = 10
        max_length = 50  # Adjust the maximum length as per your preference
        temperature = 0.7

        # Generate card names
        for _ in range(num_sequences):
            input_text = "start_token"  # Add a start token to prompt the model
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            generated_card_name = input_text

            # Generate output
            for _ in range(max_length):
                output = model.generate(
                    input_ids,
                    max_length=max_length + len(input_ids[0]),
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )

                generated_token = tokenizer.decode(output[0, len(input_ids[0]):], skip_special_tokens=True)

                generated_card_name += generated_token
                input_ids = tokenizer.encode(generated_card_name, return_tensors="pt").to(device)

            print("Generated Card Name:", generated_card_name)

except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
