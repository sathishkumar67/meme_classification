import torch


class MemeClassifier(torch.nn.Module):
    def __init__(self, img_encoder, text_encoder, num_classes=2):
        super().__init__()

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.classifier = torch.nn.Linear(512+768, num_classes)

    def forward(self, image, input_ids, attention_mask):
        # Encode the image

        image_features = self.img_encoder.encode_image(image)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # aggregate text features along the sequence dimension
        text_features = text_features.last_hidden_state[:, 0, :]

        # concatenate image and text features
        features = torch.cat((image_features, text_features), dim=1)

        # classify
        logits = self.classifier(features)
        return logits