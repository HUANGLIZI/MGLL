from sympy import true
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from .tokenizer import Tokenizer
from .bert import BertModel


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tensor_list):
        """
        tensor_list: a list of tensor with shape B x D
        """
        
        shapes = [t.shape for t in tensor_list]
        if len(set(shapes)) > 1:
            raise ValueError("All tensors in the list must have the same shape (B, D)")
        
        x = torch.stack(tensor_list, dim=1)

        # print(key.shape)
        # key = key.unsqueeze(1)  # B x 1 x D
        # print(key.shape)
        query_proj = self.query_proj(x)  
        key_proj = self.key_proj(x)        
        value_proj = self.value_proj(x)  

        # print(query_proj.shape, key_proj.shape, value_proj.shape)
        attention_scores = torch.bmm(query_proj, key_proj.transpose(1, 2))  # B x N x N
        attention_weights = self.softmax(attention_scores)  # B x N x N
        attended_values = torch.bmm(attention_weights.transpose(1, 2), value_proj)  # B x N x D
        # attended_values = attended_values.squeeze(1)  # B x D

        # FFN
        x = self.norm1(attended_values)
        x = x + self.ffn(x)
        x = self.norm2(x)
        
        processed_list = torch.unbind(x, dim=1)
        
        return processed_list

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key):
        """
        query: B x N x D
        key: B x D
        """
        # Cross Attention
        # print(key.shape)
        key = key.unsqueeze(1)  # B x 1 x D
        # print(key.shape)
        query_proj = self.query_proj(query)  
        key_proj = self.key_proj(key)        
        value_proj = self.value_proj(query)  

        # print(query_proj.shape, key_proj.shape, value_proj.shape)
        attention_scores = torch.bmm(query_proj, key_proj.transpose(1, 2))  # B x N x 1
        attention_weights = self.softmax(attention_scores)  # B x N x 1
        attended_values = torch.bmm(attention_weights.transpose(1, 2), value_proj)  # B x 1 x D
        attended_values = attended_values.squeeze(1)  # B x D

        # FFN
        x = self.norm1(attended_values)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class CLIPBaseline(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, visual_encoder, text_encoder, llama_tokenizer_path):
        super().__init__()
        
        self.logit_scale_init_value = 0.07
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        
        if 'clip' in visual_encoder or 'clip' in text_encoder:
            self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', force_custom_text=True)
            print(self.clip)
        
        if visual_encoder == 'clip_vit':
            print(self.clip.visual)
            self.clip.visual.output_tokens = True
            print('self.clip.visual.output_tokens: ', self.clip.visual.output_tokens)
        else:
            raise NotImplementedError(f"Visual encoder {visual_encoder} not implemented")
        
        if text_encoder == 'llama':
            self.tokenizer = Tokenizer(model_path=llama_tokenizer_path)
            self.context_length = 192
            self.adapter_text = nn.Linear(192, 768)
            self.text_linear = nn.Linear(768, 768)
        elif text_encoder == 'clip_text':
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
            self.context_length = 77
            print(self.clip.text)
            self.clip.text.output_tokens = True
            print('self.clip.text.output_tokens: ', self.clip.text.output_tokens)
        elif text_encoder == 'bert':
            self.text_model = BertModel(bert_type="emilyalsentzer/Bio_ClinicalBERT", proj_dim=768, proj_bias=False, projection=True, norm=True)
            self.tokenizer = self.text_model.tokenize
        elif text_encoder == 'clip_text_bert' or text_encoder == 'bert_clip_text':
            self.tokenizer_clip = open_clip.get_tokenizer('ViT-L-14')
            self.context_length = 77
            print(self.clip.text)
            self.clip.text.output_tokens = True
            print('self.clip.text.output_tokens: ', self.clip.text.output_tokens)
            self.text_model = BertModel(bert_type="emilyalsentzer/Bio_ClinicalBERT", proj_dim=768, proj_bias=False, projection=True, norm=True)
            self.tokenizer_bert = self.text_model.tokenize
        else:
            raise NotImplementedError(f"Text encoder {text_encoder} not implemented")
            
        self.text2visual = CrossAttention(768)
        self.visual2text = CrossAttention(768)
        
        self.transforms = T.Resize(size = (224,224))

        self.criterion_ab = torch.nn.CrossEntropyLoss()

        self.set_default_trainability()
    

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def ce_loss(self, pred_logit, ref):
        ce_loss = F.cross_entropy(pred_logit, ref)
        return ce_loss
    
    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            # trainable[name] = para
            if name.startswith("llama."):
                if 'norm' in name or 'bias' in name or 'lora' in name:
                    trainable[name] = para
            else:
                trainable[name] = para
        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True
    
    def clip_encode_image(self, x):
        # from CLIP
        image_encoder = self.clip.visual
        x_global, x_local = image_encoder(x)
        
        return x_global, x_local
        
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        # BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_per_text, target_pseudo)
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)  
        return (image_loss+caption_loss)/2# + BCE_loss

    def contrastive_accuracy(self, similarity_matrix, labels):
        #similarity_matrix = torch.matmul(features_i, features_j.T)
        similarity_matrix = torch.sigmoid(similarity_matrix)
        thread = 0.5
        positive_pred = (similarity_matrix > thread).float()
        # Create positive and negative masks
        positive_mask = (labels > 0).float()
        # Evaluate positive and negative pairs
        diff = (positive_pred - positive_mask).abs()
        # Compute accuracy
        accuracy = (diff < 0.5).float().mean()
        return accuracy
    
    def evaluate_mse(self, similarity_matrix, labels):
        similarity_matrix = torch.softmax(similarity_matrix, dim=1)
        mse = torch.sum((similarity_matrix - labels) ** 2)
        return mse
    
    def forward(self, imgs, Keyword_list, desc_list): # , modality_list):
        
        Keyword_tensor_list, target_list, desc_target_list, attn_mask_list, desc_tensor_list, desc_attn_mask_list = [], [], [], [], [], []
        # modality_target_list, modality_tensor_list, modality_attn_mask_list = [], [], []
        for i, Keyword_group in enumerate(Keyword_list):
            desc_group = desc_list[i]
            # modality_group = modality_list[i]
            if self.text_encoder == 'clip_text':
                token = self.tokenizer(Keyword_group, context_length=self.context_length)
                Keyword_tensor = torch.tensor(token, dtype=torch.int64).to(imgs.device)
                Keyword_tensor_list.append(Keyword_tensor)
                
                desc_token = self.tokenizer(desc_group, context_length=self.context_length)
                desc_tensor = torch.tensor(desc_token, dtype=torch.int64).to(imgs.device)
                desc_tensor_list.append(desc_tensor)
                
            elif self.text_encoder == 'bert':
                text_tokens = self.tokenizer(Keyword_group)
                input_ids = text_tokens["input_ids"].to(imgs.device).to(torch.long)
                attention_mask = text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                Keyword_tensor_list.append(input_ids)
                attn_mask_list.append(attention_mask)
                
                desc_text_tokens = self.tokenizer(desc_group)
                desc_input_ids = desc_text_tokens["input_ids"].to(imgs.device).to(torch.long)
                desc_attention_mask = desc_text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                desc_tensor_list.append(desc_input_ids)
                desc_attn_mask_list.append(desc_attention_mask)
                
                # modality_text_tokens = self.tokenizer(modality_group)
                # modality_input_ids = modality_text_tokens["input_ids"].to(imgs.device).to(torch.long)
                # modality_attention_mask = modality_text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                # modality_tensor_list.append(modality_input_ids)
                # modality_attn_mask_list.append(modality_attention_mask)
                
            elif self.text_encoder == 'clip_text_bert':
                token = self.tokenizer_clip(Keyword_group, context_length=self.context_length)
                Keyword_tensor = torch.tensor(token, dtype=torch.int64).to(imgs.device)
                Keyword_tensor_list.append(Keyword_tensor)
                
                desc_text_tokens = self.tokenizer_bert(desc_group)
                desc_input_ids = desc_text_tokens["input_ids"].to(imgs.device).to(torch.long)
                desc_attention_mask = desc_text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                desc_tensor_list.append(desc_input_ids)
                desc_attn_mask_list.append(desc_attention_mask)
                
            elif self.text_encoder == 'bert_clip_text':
                text_tokens = self.tokenizer_bert(Keyword_group)
                input_ids = text_tokens["input_ids"].to(imgs.device).to(torch.long)
                attention_mask = text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                Keyword_tensor_list.append(input_ids)
                attn_mask_list.append(attention_mask)
                
                desc_token = self.tokenizer_clip(desc_group, context_length=self.context_length)
                desc_tensor = torch.tensor(desc_token, dtype=torch.int64).to(imgs.device)
                desc_tensor_list.append(desc_tensor)
                
            else:
                raise NotImplementedError(f"Text encoder {self.text_encoder} not implemented")

        for i in range(len(Keyword_tensor_list)):
            if self.text_encoder == 'llama':
                text_feats = self.adapter_text(Keyword_tensor_list[i])
                text_feats = F.normalize(text_feats)
                text_global = self.text_linear(text_feats)
                text_local = text_global.unsqueeze(1)
            elif self.text_encoder == 'clip_text':
                text_global, text_local = self.clip.text(Keyword_tensor_list[i])
                desc_global, desc_local = self.clip.text(desc_tensor_list[i])
            elif self.text_encoder == 'bert':
                text_global = self.text_model(Keyword_tensor_list[i], attn_mask_list[i])
                desc_global = self.text_model(desc_tensor_list[i], desc_attn_mask_list[i])
                # modality_global = self.text_model(modality_tensor_list[i], modality_attn_mask_list[i])
            elif self.text_encoder == 'clip_text_bert':
                text_global, text_local = self.clip.text(Keyword_tensor_list[i])
                desc_global = self.text_model(desc_tensor_list[i], desc_attn_mask_list[i])
            elif self.text_encoder == 'bert_clip_text':
                text_global = self.text_model(Keyword_tensor_list[i], attn_mask_list[i])
                desc_global, desc_local = self.clip.text(desc_tensor_list[i])
            else:
                raise NotImplementedError(f"Text encoder {self.text_encoder} not implemented")

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in Keyword_group] for iiDesc in Keyword_group], np.float32)
            coocurrence_normalized = coocurrence # / coocurrence.sum(axis=-1, keepdims=True)
            target = torch.tensor(coocurrence_normalized, device=imgs.device, dtype=torch.float32)
            target_list.append(target)
            
            desc_coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in desc_group] for iiDesc in desc_group], np.float32)
            desc_coocurrence_normalized = desc_coocurrence # / desc_coocurrence.sum(axis=-1, keepdims=True)
            desc_target = torch.tensor(desc_coocurrence_normalized, device=imgs.device, dtype=torch.float32)
            desc_target_list.append(desc_target)
            
            # modality_coocurrence = np.array(
            #     [[iDesc == iiDesc for iDesc in modality_group] for iiDesc in modality_group], np.float32)
            # modality_coocurrence_normalized = modality_coocurrence / modality_coocurrence.sum(axis=-1, keepdims=True)
            # modality_target = torch.tensor(modality_coocurrence_normalized, device=imgs.device, dtype=torch.float32)
            # modality_target_list.append(modality_target)
            
        if self.visual_encoder == 'clip_vit':
            image_global, image_local = self.clip.visual(self.transforms(imgs))
            # image_local = self.visual_local(image_local)
        else:
            raise NotImplementedError(f"Visual encoder {self.visual_encoder} not implemented")
        
        clip_loss = 0
        clip_score = 0

        for i in range(len(target_list)):
            logits_per_text = self.compute_logits(image_global, text_global)
            clip_loss += self.softce_clip_loss(logits_per_text, target_list[i])
            clip_score += self.contrastive_accuracy(logits_per_text.T, target_list[i])

        return clip_loss, clip_score


class MultiGranularModule(nn.Module):

    def __init__(self, visual_encoder, text_encoder, llama_tokenizer_path):
        super().__init__()
        
        self.logit_scale_init_value = 0.07
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        
        if 'clip' in visual_encoder or 'clip' in text_encoder:
            self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', force_custom_text=True)
            print(self.clip)
        
        if visual_encoder == 'clip_vit':
            print(self.clip.visual)
            self.clip.visual.output_tokens = True
            print('self.clip.visual.output_tokens: ', self.clip.visual.output_tokens)
        else:
            raise NotImplementedError(f"Visual encoder {visual_encoder} not implemented")
        
        if text_encoder == 'llama':
            self.tokenizer = Tokenizer(model_path=llama_tokenizer_path)
            self.context_length = 192
            self.adapter_text = nn.Linear(192, 768)
            self.text_linear = nn.Linear(768, 768)
        elif text_encoder == 'clip_text':
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
            self.context_length = 77
            print(self.clip.text)
            self.clip.text.output_tokens = True
            print('self.clip.text.output_tokens: ', self.clip.text.output_tokens)
        elif text_encoder == 'bert':
            self.text_model = BertModel(bert_type="emilyalsentzer/Bio_ClinicalBERT", proj_dim=768, proj_bias=False, projection=True, norm=True)
            self.tokenizer = self.text_model.tokenize
        elif text_encoder == 'clip_text_bert' or text_encoder == 'bert_clip_text':
            self.tokenizer_clip = open_clip.get_tokenizer('ViT-L-14')
            self.context_length = 77
            print(self.clip.text)
            self.clip.text.output_tokens = True
            print('self.clip.text.output_tokens: ', self.clip.text.output_tokens)
            self.text_model = BertModel(bert_type="emilyalsentzer/Bio_ClinicalBERT", proj_dim=768, proj_bias=False, projection=True, norm=True)
            self.tokenizer_bert = self.text_model.tokenize
        else:
            raise NotImplementedError(f"Text encoder {text_encoder} not implemented")
            
        self.transforms = T.Resize(size = (224,224))
        self.set_default_trainability()

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def ce_loss(self, pred_logit, ref):
        ce_loss = F.cross_entropy(pred_logit, ref)
        return ce_loss
    
    def kl_loss_2(self, pred_logit_1, pred_logit_2):
        M = 0.5 * (pred_logit_1+ pred_logit_2)
        kl_loss = 0.5 * F.kl_div(pred_logit_1.log(), M, reduction='batchmean')+0.5 * F.kl_div(pred_logit_2.log(), M, reduction='batchmean')
        return kl_loss
    
    def kl_loss_3(self, pred_logit_1, pred_logit_2, pred_logit_3):
        M = (pred_logit_1 + pred_logit_2 + pred_logit_3)/3.0
        kl_loss = (F.kl_div(pred_logit_1.log(), M, reduction='batchmean')+F.kl_div(pred_logit_2.log(), M, reduction='batchmean')+F.kl_div(pred_logit_3.log(), M, reduction='batchmean'))/3.0
        return kl_loss

    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            # trainable[name] = para
            if name.startswith("llama."):
                if 'norm' in name or 'bias' in name or 'lora' in name:
                    trainable[name] = para
            else:
                trainable[name] = para
        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True
        
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_per_text, target_pseudo)
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)  
        return (image_loss+caption_loss)/4.0 + BCE_loss

    def contrastive_accuracy(self, similarity_matrix, labels):
        #similarity_matrix = torch.matmul(features_i, features_j.T)
        similarity_matrix = torch.sigmoid(similarity_matrix)
        thread = 0.5
        positive_pred = (similarity_matrix > thread).float()
        # Create positive and negative masks
        positive_mask = (labels > 0).float()
        # Evaluate positive and negative pairs
        diff = (positive_pred - positive_mask).abs()
        # Compute accuracy
        accuracy = (diff < 0.5).float().mean()
        return accuracy
    
    def evaluate_mse(self, similarity_matrix, labels):
        similarity_matrix = torch.softmax(similarity_matrix, dim=1)
        mse = torch.sum((similarity_matrix - labels) ** 2)
        return mse
    
    def forward(self, imgs, Keyword_list, desc_list): # , modality_list):
        
        Keyword_tensor_list, target_list, desc_target_list, attn_mask_list, desc_tensor_list, desc_attn_mask_list = [], [], [], [], [], []
        # modality_target_list, modality_tensor_list, modality_attn_mask_list = [], [], []
        for i, Keyword_group in enumerate(Keyword_list):
            desc_group = desc_list[i]
            # modality_group = modality_list[i]
            if self.text_encoder == 'clip_text':
                token = self.tokenizer(Keyword_group, context_length=self.context_length)
                Keyword_tensor = torch.tensor(token, dtype=torch.int64).to(imgs.device)
                Keyword_tensor_list.append(Keyword_tensor)
                
                desc_token = self.tokenizer(desc_group, context_length=self.context_length)
                desc_tensor = torch.tensor(desc_token, dtype=torch.int64).to(imgs.device)
                desc_tensor_list.append(desc_tensor)
                
            elif self.text_encoder == 'bert':
                text_tokens = self.tokenizer(Keyword_group)
                input_ids = text_tokens["input_ids"].to(imgs.device).to(torch.long)
                attention_mask = text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                Keyword_tensor_list.append(input_ids)
                attn_mask_list.append(attention_mask)
                
                desc_text_tokens = self.tokenizer(desc_group)
                desc_input_ids = desc_text_tokens["input_ids"].to(imgs.device).to(torch.long)
                desc_attention_mask = desc_text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                desc_tensor_list.append(desc_input_ids)
                desc_attn_mask_list.append(desc_attention_mask)
                
                # modality_text_tokens = self.tokenizer(modality_group)
                # modality_input_ids = modality_text_tokens["input_ids"].to(imgs.device).to(torch.long)
                # modality_attention_mask = modality_text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                # modality_tensor_list.append(modality_input_ids)
                # modality_attn_mask_list.append(modality_attention_mask)
                
            elif self.text_encoder == 'clip_text_bert':
                token = self.tokenizer_clip(Keyword_group, context_length=self.context_length)
                Keyword_tensor = torch.tensor(token, dtype=torch.int64).to(imgs.device)
                Keyword_tensor_list.append(Keyword_tensor)
                
                desc_text_tokens = self.tokenizer_bert(desc_group)
                desc_input_ids = desc_text_tokens["input_ids"].to(imgs.device).to(torch.long)
                desc_attention_mask = desc_text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                desc_tensor_list.append(desc_input_ids)
                desc_attn_mask_list.append(desc_attention_mask)
                
            elif self.text_encoder == 'bert_clip_text':
                text_tokens = self.tokenizer_bert(Keyword_group)
                input_ids = text_tokens["input_ids"].to(imgs.device).to(torch.long)
                attention_mask = text_tokens["attention_mask"].to(imgs.device).to(torch.long)
                Keyword_tensor_list.append(input_ids)
                attn_mask_list.append(attention_mask)
                
                desc_token = self.tokenizer_clip(desc_group, context_length=self.context_length)
                desc_tensor = torch.tensor(desc_token, dtype=torch.int64).to(imgs.device)
                desc_tensor_list.append(desc_tensor)
                
            else:
                raise NotImplementedError(f"Text encoder {self.text_encoder} not implemented")

        for i in range(len(Keyword_tensor_list)):
            if self.text_encoder == 'llama':
                text_feats = self.adapter_text(Keyword_tensor_list[i])
                text_feats = F.normalize(text_feats)
                text_global = self.text_linear(text_feats)
                text_local = text_global.unsqueeze(1)
            elif self.text_encoder == 'clip_text':
                text_global, text_local = self.clip.text(Keyword_tensor_list[i])
                desc_global, desc_local = self.clip.text(desc_tensor_list[i])
            elif self.text_encoder == 'bert':
                text_global = self.text_model(Keyword_tensor_list[i], attn_mask_list[i])
                desc_global = self.text_model(desc_tensor_list[i], desc_attn_mask_list[i])
                # modality_global = self.text_model(modality_tensor_list[i], modality_attn_mask_list[i])
            elif self.text_encoder == 'clip_text_bert':
                text_global, text_local = self.clip.text(Keyword_tensor_list[i])
                desc_global = self.text_model(desc_tensor_list[i], desc_attn_mask_list[i])
            elif self.text_encoder == 'bert_clip_text':
                text_global = self.text_model(Keyword_tensor_list[i], attn_mask_list[i])
                desc_global, desc_local = self.clip.text(desc_tensor_list[i])
            else:
                raise NotImplementedError(f"Text encoder {self.text_encoder} not implemented")

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in Keyword_group] for iiDesc in Keyword_group], np.float32)
            coocurrence_normalized = coocurrence / coocurrence.sum(axis=-1, keepdims=True)
            target = torch.tensor(coocurrence_normalized, device=imgs.device, dtype=torch.float32)
            target_list.append(target)
            
            desc_coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in desc_group] for iiDesc in desc_group], np.float32)
            desc_coocurrence_normalized = desc_coocurrence / desc_coocurrence.sum(axis=-1, keepdims=True)
            desc_target = torch.tensor(desc_coocurrence_normalized, device=imgs.device, dtype=torch.float32)
            desc_target_list.append(desc_target)
            
            # modality_coocurrence = np.array(
            #     [[iDesc == iiDesc for iDesc in modality_group] for iiDesc in modality_group], np.float32)
            # modality_coocurrence_normalized = modality_coocurrence / modality_coocurrence.sum(axis=-1, keepdims=True)
            # modality_target = torch.tensor(modality_coocurrence_normalized, device=imgs.device, dtype=torch.float32)
            # modality_target_list.append(modality_target)
            
        if self.visual_encoder == 'clip_vit':
            image_global, image_local = self.clip.visual(self.transforms(imgs))
            # image_local = self.visual_local(image_local)
        else:
            raise NotImplementedError(f"Visual encoder {self.visual_encoder} not implemented")
        
        clip_loss = 0
        clip_score = 0

        for i in range(len(target_list)):
            logits_per_text = self.compute_logits(image_global, text_global)
            clip_loss += self.softce_clip_loss(logits_per_text, target_list[i])
            logits_per_desc = self.compute_logits(image_global, desc_global)
            clip_loss += self.softce_clip_loss(logits_per_desc, desc_target_list[i])
            # logits_per_modality = self.compute_logits(image_global, modality_global)
            # clip_loss += self.softce_clip_loss(logits_per_modality, modality_target_list[i])
            # distill_loss = self.kl_loss_3(logits_per_text, logits_per_desc, logits_per_modality)*0.5
            distill_loss = self.kl_loss_2(logits_per_text, logits_per_desc)*0.5
            if not torch.isnan(distill_loss) and not torch.isinf(distill_loss):
                clip_loss += distill_loss
            clip_score += self.contrastive_accuracy(logits_per_text.T, target_list[i])
            # clip_score += self.evaluate_mse(logits_per_text, target_list[i])

        return clip_loss, clip_score
