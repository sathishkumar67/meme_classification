from typing import Tuple
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose
from transformers import PreTrainedTokenizer


class MemeDataset(Dataset):
    def __init__(self, image_folder_path: str, caption_file: str, image_transform: Compose=None, tokenizer: PreTrainedTokenizer=None):
        """
        Args:
            image_folder_path (str): Path to the image folder.
            caption_file (str): Path to the CSV file containing captions.
            image_transform (Callable): Transformations to apply to images.
            """
        self.image_folder_path = image_folder_path
        self.captions_df = pd.read_csv(caption_file, index_col=0)
        self.image_dataset = datasets.ImageFolder(self.image_folder_path, transform=image_transform)
        self.tokenized_captions = tokenizer(
            self.captions_df["captions"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        )


    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx) -> Tuple:
        # Load image and label
        preprocessed_image, label = self.image_dataset[idx]

        # Load caption
        input_ids = self.tokenized_captions["input_ids"][idx]
        attention_mask = self.tokenized_captions["attention_mask"][idx]

        return preprocessed_image, label, input_ids, attention_mask