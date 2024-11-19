import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_counter_argument_model(model_path, base_model_name):
    """Load the fine-tuned model and tokenizer for counter-argument classification."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def classify_counter_hate(hate_tweet, counter_hate, model, tokenizer):
    """Classify if the counter-hate is effective against the hate tweet."""
    try:
        input_text = f"{hate_tweet} [SEP] {counter_hate}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
        
        effectiveness = "Effective" if predicted_class.item() == 1 else "Ineffective"
        return effectiveness, predicted_class.item()
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Error occurred during classification.", None

def main():
    parser = argparse.ArgumentParser(description="Counter-hate classification")
    parser.add_argument("--trained-model-dir", required=False, default="./Output",
                        help="Location of the saved trained model.")
    parser.add_argument("--base-model-name", required=False, default="allenai/longformer-base-4096",
                        help="Name of the base model used for the tokenizer.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.trained_model_dir):
        trained_model_dir = args.trained_model_dir
    else:
        raise Exception(f'Trained model directory "{args.trained_model_dir}" does not exist.')

    model_path = os.path.join(trained_model_dir, "final_trained_model")
    model, tokenizer = load_counter_argument_model(model_path, args.base_model_name)

    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        exit(1)

    test_pairs = [
        ("Messi is a racist!!!! He should get eliminated.", "Messi has shown respect for all races throughout his career. It's important to base opinions on facts, not unfounded accusations."),
        ("Football is an amazing sport that brings people together.", "You're right! Football has a unique ability to unite people from diverse backgrounds."),
        ("Avril Lavigne is stupid, she needs a punch in the face.", "Violence is never the answer. Everyone deserves respect, including artists like Avril Lavigne. Let's focus on constructive criticism instead.")
    ]

    for hate_tweet, counter_hate in test_pairs:
        print("Hate tweet:")
        print(hate_tweet)
        print("Counter-hate:")
        print(counter_hate)
        effectiveness, label = classify_counter_hate(hate_tweet, counter_hate, model, tokenizer)
        print(f"Classification: {effectiveness} (Label: {label})")
        print()

if __name__ == "__main__":
    main()