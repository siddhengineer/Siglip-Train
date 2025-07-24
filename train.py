import os
import torch
from tqdm import tqdm
from transformers import AutoModel
from data_loader import create_data_loaders
from config import OUTPUT_DIR, EPOCHS, LEARNING_RATE, MODEL_NAME # Import MODEL_NAME

def train_model(model, train_loader, val_loader, epochs, learning_rate, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], pixel_values=batch['pixel_values'], return_loss=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], pixel_values=batch['pixel_values'], return_loss=True)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train_loader, val_loader, processor = create_data_loaders()
    model = AutoModel.from_pretrained(MODEL_NAME)
    train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, OUTPUT_DIR)
    print("Training completed successfully!")