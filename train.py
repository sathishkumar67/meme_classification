import clip
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, BertModel
from dataset import MemeDataset
from model import MemeClassifier
from tqdm import tqdm

# Load the CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
img_encoder, img_preprocess = clip.load("ViT-B/16", device=device)  # You can also use "ViT-B/16" or "RN50"

text_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text_encoder = BertModel.from_pretrained("bert-base-multilingual-cased")

# Load the dataset
meme_dataset = MemeDataset(
    image_folder_path="/content/meme_classification/dataset/train",
    caption_file="/content/meme_classification/dataset/train_captions.csv",
    image_transform=img_preprocess,
    tokenizer=text_tokenizer
)

# Define split sizes
train_size = int(0.8 * len(meme_dataset))  # 80% for training
val_size = len(meme_dataset) - train_size  # Remaining 20% for validation

# Split the dataset
train_dataset, val_dataset = random_split(meme_dataset, [train_size, val_size])

# Create DataLoaders for train and val
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model
model = MemeClassifier(img_encoder, text_encoder)
model.to(device)

num_epochs = 5
# Define the optimizer and loss function
optimizer = torch.optim.Adam(text_encoder.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        images, labels, input_ids, attention_mask = batch
        images = images.to(device)
        labels = labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        output = model(images, input_ids, attention_mask)
        loss = criterion(output, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss
        epoch_loss += loss.item()

        # Update tqdm progress bar with loss
        progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Free memory for next iteration
        del images, labels, input_ids, attention_mask, output, loss

    # Log epoch loss
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(train_loader):.4f}")


# Validation loop
# predict on test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in val_loader:
        images, labels, input_ids, attention_mask = batch
        images = images.to(device)
        labels = labels.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        output = model(images, input_ids, attention_mask)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total}")
    del images, labels, input_ids, attention_mask, output