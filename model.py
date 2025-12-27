import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import config

print("--> Đang tải PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
text_encoder_base = AutoModel.from_pretrained("vinai/phobert-base-v2")


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected  # Residual
        x = self.layer_norm(x)
        return x


class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embedding_dim=256):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.vis_project = ProjectionHead(config.IMAGE_EMBED_DIM, embedding_dim)
        self.text_project = ProjectionHead(config.TEXT_EMBED_DIM, embedding_dim)


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, text_sentences):
        image_features = self.vision_encoder(images)
        image_embeddings = self.vis_project(image_features)

        inputs = tokenizer(
            text_sentences,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        ).to(config.DEVICE)

        text_outputs = self.text_encoder(**inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_embeddings = self.text_project(text_features)


        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)


        logit_scale = 100.0
        logits = (text_embeddings @ image_embeddings.T) * logit_scale


        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T


        targets = F.softmax((images_similarity + texts_similarity) / 2.0 * logit_scale, dim=-1)

        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0

        return loss.mean()