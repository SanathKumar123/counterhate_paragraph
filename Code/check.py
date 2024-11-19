import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os

def load_counter_argument_model(model_path):
    """Load the fine-tuned model for counter-argument classification."""
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        # print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        # print(f"Error loading counter-argument model: {e}")
        return None, None
    return model, tokenizer

def classify_counter_hate(hate_tweet, counter_hate, model, tokenizer, device):
    """Classify if the counter-hate is effective against the hate tweet."""
    try:
        input_text = f"Hate tweet: {hate_tweet}\nCounter-hate: {counter_hate}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
        
        effectiveness = "Effective" if predicted_class.item() == 1 else "Ineffective"
        return effectiveness, predicted_class.item()
    except Exception as e:
        # print(f"Error during classification: {e}")
        return "Error occurred during classification.", None

if __name__ == "__main__":
    current_dir = os.getcwd()
    counter_argument_model_path = os.path.join(current_dir, "Output", "final_trained_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"Loading model from: {counter_argument_model_path}")
    # print(f"Using device: {device}")

    model, tokenizer = load_counter_argument_model(counter_argument_model_path)

    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        exit(1)

    test_pairs = [
        ("Messi is a racist!!!! He should get eliminated.", "Messi has shown respect for all races throughout his career. It's important to base opinions on facts, not unfounded accusations."),
        ("Football is an amazing sport that brings people together.", "You're right! Football has a unique ability to unite people from diverse backgrounds."),
        ("Avril Lavigne is stupid, she needs a punch in the face.", "Violence is never the answer. Everyone deserves respect, including artists like Avril Lavigne. Let's focus on constructive criticism instead.")
    ]

    for hate_tweet, counter_hate in test_pairs:
        print("hate_tweet")
        print(hate_tweet)
        print("Counter-hate:")
        print({counter_hate})
        effectiveness, label = classify_counter_hate(hate_tweet, counter_hate, model, tokenizer, device)
        print(f"Classification:") 
        print({effectiveness}, {label})