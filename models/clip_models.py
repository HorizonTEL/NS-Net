from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
from .loralib.utils import apply_lora, mark_only_lora_as_trainable


class ProjectionHead(nn.Module):
    """
    Projection head as used in SimCLR to map features to a space where contrastive loss is applied.
    This can help improve the quality of the learned representations.
    """

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class CLIPModel(nn.Module):
    def __init__(self, opt, num_classes=1):
        super(CLIPModel, self).__init__()

        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.clip_model, self.preprocess = clip.load(self.opt.backbone, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        list_lora_layers = apply_lora(self.opt, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)

        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, text):
        features = self.clip_model.encode_image(x)
        text_tokens = clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)

        U, S, Vt = torch.linalg.svd(text_features, full_matrices=True)
        rank = (S > 1e-10).sum().item()
        V_null = Vt[rank:, :].T
        projection_matrix = V_null @ V_null.T
        features_artifact = features @ projection_matrix

        labels = self.fc(features_artifact)
        return {'label': labels, 'features_artifact': features_artifact}