import argparse
import torch
import os
from transformers import (
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import time
import datetime
from torchmetrics.classification import BinaryAccuracy

# Hyperparameter values to experiment with
EPOCHS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
MAX_PATIENCE = 2
ACCUMULATE_GRADIENTS = 2  # Gradients are accumulated for every two batches

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(train_dataloader, validation_dataloader, model, scheduler, optimizer, writer, output_dir):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    training_stats = []
    total_t0 = time.time()
    best_score = float("inf")
    epochs_without_improvement = 0
    accuracy_metric = BinaryAccuracy().to(DEVICE)

    for epoch_i in range(EPOCHS):
        print(f"\n======== Epoch {epoch_i + 1} / {EPOCHS} ========")
        print("Training...")

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step:>5} of {len(train_dataloader):>5}. Elapsed: {elapsed}.")

            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()

            loss.backward()

            # Gradient accumulation
            if (step + 1) % ACCUMULATE_GRADIENTS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
                optimizer.step()
                scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")

        # TensorBoard logging
        writer.add_scalar('Training Loss', avg_train_loss, epoch_i)

        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print(f"  Accuracy: {avg_val_accuracy:.2f}")

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        # TensorBoard logging
        writer.add_scalar('Validation Loss', avg_val_loss, epoch_i)
        writer.add_scalar('Validation Accuracy', avg_val_accuracy, epoch_i)

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

        if avg_val_loss < best_score:
            print("Saving the best model...")
            best_score = avg_val_loss
            # model.save_pretrained(os.path.join(output_dir, "best_trained_model"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement > MAX_PATIENCE:
            print("Early stopping...")
            break

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=False,
        default="./Dataloaders",
        help="Location of data files.",
    )
    parser.add_argument(
        "--level",
        required=True,
        help="The level to work with, either 'paragraph' or 'article'.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./Output",
        help="Output directory to save the trained model.",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    if args.level not in ["paragraph", "article"]:
        raise Exception("Level must equal either 'paragraph' or 'article'.")
    else:
        level = args.level
        EXP = ["tweet", level]

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_dataloader = torch.load(os.path.join(data_dir, "train.pth"))
    validation_dataloader = torch.load(os.path.join(data_dir, "valid.pth"))

    if level == "paragraph":
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2,  # Binary classification
            output_attentions=False,
            output_hidden_states=False,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            attention_window=512,
        )

    # Modify model layers (output features and activation)
    model.classifier.dense.out_features = 128
    model.classifier.dense.add_module("Activation", torch.nn.ReLU())
    model.classifier.dense.add_module("Dropout", torch.nn.Dropout(0.3))  # Add Dropout
    model.classifier.out_proj.in_features = 128
    model.classifier.out_proj.add_module("Activation", torch.nn.Softmax(dim=-1))

    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,  # Regularization
    )

    total_steps = len(train_dataloader) * EPOCHS

    # Using cosine scheduler for learning rate
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,  # Adjust warmup steps as needed
        num_training_steps=total_steps,
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    train(train_dataloader, validation_dataloader, model, scheduler, optimizer, writer, output_dir)
    model.save_pretrained(os.path.join(output_dir, "final_trained_model"))

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
