import timm
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import vit_l_16, ViT_L_16_Weights
import timm.models.vision_transformer


class DownstreamModule(nn.Module):
    """ Downstream module
    """
    def __init__(self, visual_encoder, task, num_classes, is_probe=False, default_pretrain=None, return_loss=True): #, text_encoder, llama_tokenizer_path, knn=False):
        super().__init__()

        self.visual_encoder = visual_encoder
        self.task = task
        self.is_probe = is_probe
        self.return_loss = return_loss
        
        if 'clip' in visual_encoder:
            if default_pretrain == "None":
                default_pretrain = None
            self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=default_pretrain, force_custom_text=True)
        
        if visual_encoder == 'clip_vit':
            self.classifier = nn.Linear(768, num_classes)
        elif visual_encoder == 'vit_l_16':
            if default_pretrain == "None":
                weights = None
            elif default_pretrain == "imagenet1kv1":
                weights = ViT_L_16_Weights.IMAGENET1K_V1
            self.vit_l_16 = vit_l_16(weights=weights)
            self.vit_l_16.heads.head = nn.Linear(self.vit_l_16.heads.head.in_features, num_classes)
            nn.init.xavier_uniform_(self.vit_l_16.heads.head.weight)
            nn.init.zeros_(self.vit_l_16.heads.head.bias)
        else:
            raise NotImplementedError(f"Visual encoder {visual_encoder} not implemented")
        
        self.transforms = T.Resize(size = (224,224))

        if task == 'multilabel':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.set_default_trainability()

    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            # trainable[name] = para
            if self.is_probe:
                if name.startswith("classifier"):
                    trainable[name] = para
            else:
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                # elif name.startswith("eva02."):
                #     if 'adapter' in name:
                #         trainable[name] = para
                # elif name.startswith("LViT."):
                #     trainable[name] = para
                else:
                    trainable[name] = para
        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True

    def forward(self, imgs, labels=None):
        
        if self.visual_encoder == 'clip_vit':
            image_global = self.clip.visual(self.transforms(imgs))
            # image_global = F.sigmoid(image_global)
            logits = self.classifier(image_global)
        elif self.visual_encoder == 'vit_l_16':
            logits = self.vit_l_16(self.transforms(imgs))
        else:
            raise NotImplementedError(f"Visual encoder {self.visual_encoder} not implemented")
        
        if self.return_loss:
            loss = self.criterion(logits, labels)            
            return logits, loss 
        else:
            return logits
    