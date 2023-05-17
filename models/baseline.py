
import torch
from torch import nn
import torch.nn.functional as F
# from Nets import *
from utils.options import args_parser
from models.Nets import MLP
from utils.options import args_parser
# Import necessary PyTorch modules

from torchvision.models import vit_b_16, ViT_B_16_Weights, VisionTransformer
from transformers import BertTokenizer, BertModel, BertConfig, ViltModel
from sentence_transformers import SentenceTransformer, util
import timm



args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


# Create a simple ViT model
class ImageEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.vit = vit_b_16(pretrained=True)

  def forward(self, x):
      # self.vit.eval()
      x = self.vit._process_input(x)
      n = x.shape[0]

      batch_class_token = self.vit.class_token.expand(n,-1,-1)
      x = torch.cat([batch_class_token, x], dim=1)
      x = self.vit.encoder(x)

      return x.mean(dim=1)

# Create a BERT model
# bert_config = BertConfig(hidden_size=256)

class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.bert = BertModel.from_pretrained('bert-base-uncased')

  def forward(self, x):

    output = self.bert(x).pooler_output
    return output

# Create a model that concatenates the image and text encoders

class shared_projector(nn.Module):
    def __init__(self, image_encoder=ImageEncoder(), text_encoder=TextEncoder()):
        super(shared_projector, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, image, text):
        image_embedding = self.image_encoder(image)
        text_embedding = self.text_encoder(text)
        concatenated_embedding = torch.cat((image_embedding, text_embedding), dim=1)
        return concatenated_embedding


class FedViT(nn.Module):
    def __init__(self, args, image_size):
        super(FedViT, self).__init__()
        self.args = args
        if self.args.pretrain:
            self.vit = vit_b_16(pretrained=True)
            # for p in self.parameters():
            #     p.requires_grad = False
            self.classifier = MLP(1000, 768, 512, self.args.num_classes)
        else:
            self.vit = VisionTransformer(
                image_size=image_size,
                patch_size=16,
                num_heads=4,
                num_layers=2,
                mlp_dim=256,
                hidden_dim= 256,
                num_classes= self.args.num_classes)

    def forward(self, image):
        # pretrained: batch size * 224 * 224
        if self.args.pretrain:

            x = self.vit(image) # (batch size, 1000)
            output = self.classifier(x) # (batch size, num_classes)
        else: # non_pretrained: batch size * 256 * 256
            output = self.vit(image) # (batch size, num_classes)

        return output



class FedBERT(nn.Module):
    def __init__(self, args):
        super(FedBERT, self).__init__()
        self.args = args
        self.config = BertConfig(hidden_size = 256, num_hidden_layers = 4, num_attention_heads = 4)
        if self.args.pretrain:

            self.bert = BertModel.from_pretrained('bert-base-uncased')
            # for p in self.parameters():
            #     p.requires_grad = False
            self.l2 = torch.nn.Dropout(0.3)
       
            self.l3 = torch.nn.Linear(768, self.args.num_classes)

            # self.classifier = MLP(768, 512, 256, self.args.num_classes)

        else:
            self.bert = BertModel(self.config)
            self.fc = nn.Linear(256,self.args.num_classes)

    def forward(self, text, attention_mask=None, token_id=None):

        if self.args.pretrain:
            # with torch.no_grad():
            embedding = self.bert(text, attention_mask=attention_mask, token_type_ids=token_id).pooler_output
            # embedding = embedding.mean(dim=1)

            output = self.l2(embedding)
            output = self.l3(output)
        else:
            embedding = self.bert(text).pooler_output #(batch size, embedding_dim)
            output = self.fc(embedding)

        return output



class FedRn50(nn.Module):
    def __init__(self, args):
        super(FedRn50, self).__init__()
        self.args = args

        self.model = timm.create_model('cspresnet50', pretrained=self.args.pretrain, num_classes=self.args.num_classes)


    def forward(self, x):
        return self.model(x)




class FedCLIP(nn.Module):
    def __init__(self, temperature=0.07, num_classes=args.num_classes):
        super(FedCLIP, self).__init__()

        # self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = vit_b_16(pretrained=True)
        self.fc_vit_trans = nn.Linear(1000, 768)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 256)
        )
        self.fc = nn.Linear(512, num_classes)
        # temperature
        self.temperature = torch.tensor(temperature, requires_grad=True)
        
        # hyperpara for loss weights
        self.alpha = 1
        self.beta = 1
        
        
    def contrastive_score(self, image_features, text_features):
    
        # # positive pair (128,256)
        # positive_pairs = torch.cat((image_features.unsqueeze(1), text_features.unsqueeze(1)), dim=1).view(-1, 256)
        # # negative pairs (4096,512)
        # num_images = image_features.shape[0]
        # num_texts = text_features.shape[0]
        # negative_pairs = torch.cat((image_features.repeat_interleave(num_texts, dim=0), text_features.repeat(num_images, 1)), dim=1)

        
        
        # # calculate similarity scores between positive and negative pairs
        # similarity_scores = torch.cat((positive_pairs, negative_pairs), dim=0)
        # similarity_scores = F.cosine_similarity(similarity_scores.unsqueeze(1), similarity_scores.unsqueeze(0), dim=2)
        # # divide by temperature
        # similarity_scores /= self.temperature
        
        # # calculate contrastive loss
        # labels = torch.zeros(num_images+num_texts, dtype=torch.long).to(images.device)
        # contrastive_loss = nn.CrossEntropyLoss()(similarity_scores, labels)
        
        
        #  # repeat image features and text features to create positive pairs
        # positive_pairs = torch.cat((image_features.repeat(1, text_features.shape[1], 1), text_features.repeat(image_features.shape[0], 1, 1)), dim=2).view(-1, 256)
        # # repeat image features and text features to create negative pairs
        # negative_pairs = torch.cat((image_features.repeat_interleave(text_features.shape[1], dim=1).view(-1, 256), text_features.repeat(image_features.shape[0], 1, 1).view(-1, 256)), dim=1)

        # # calculate similarity scores between positive and negative pairs
        # similarity_scores = torch.cat((positive_pairs, negative_pairs), dim=0)
        # similarity_scores = F.cosine_similarity(similarity_scores.unsqueeze(1), similarity_scores.unsqueeze(0), dim=2)
        # # divide by temperature
        # similarity_scores /= self.temperature
        
        
        
        # labels = torch.zeros(image_features.shape[0]*text_features.shape[1]*2, dtype=torch.long).to(images.device)
        # contrastive_loss = nn.CrossEntropyLoss()(similarity_scores, labels)
        
        logits = torch.exp(self.temperature) * torch.mm(image_features, text_features.T)
        labels = torch.arange(logits.shape[0]).to(args.device)
        
        loss_i = F.cross_entropy(logits, labels, reduction='sum')
        loss_t = F.cross_entropy(logits.T, labels, reduction='sum')
        
        return (loss_i + loss_t) / (2 * logits.shape[0])
    
    def supervised_loss(self, image_features, text_features, labels):
        
        joint_representation = torch.cat((image_features, text_features), dim=1)
        logits = self.fc(joint_representation)
        supervised_loss = F.cross_entropy(logits, labels)
        
        return supervised_loss, logits

    def encode(self, images, texts, attention_mask=None, token_id=None):
        # encode image
        # image_embedding = self.vit_model(images).last_hidden_state[:, 0]
        image_embedding = self.fc_vit_trans(self.vit_model(images))
        # print(f'image_embedding:{image_embedding.shape}')
        # encode text
        text_embedding = self.bert_model(texts,attention_mask=attention_mask,token_type_ids=token_id).pooler_output
        # print(f'text_embedding:{text_embedding.shape}')
            
        # text_embedding = self.bert_model(texts,attention_mask=attention_mask,token_type_ids=token_id)[0][:, 0, :]
        # encode both image and text again for contrastive loss
        image_features = self.projection_head(image_embedding)
        text_features = self.projection_head(text_embedding)

        # normalize the features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        return image_features, text_features
    
    def test(self, images, texts, attention_mask=None, token_type_ids=None):
        
        image_features, text_features = self.encode(images, texts, attention_mask, token_type_ids)
        # print(f'image_features:{image_features.shape}')
        # print(f'text_features:{text_features.shape}')
        
        joint_representation = torch.cat((image_features, text_features), dim=1)
        # print(f'joint_embed:{joint_representation.shape}')
        logits = self.fc(joint_representation)
        
        return logits
        
    def forward(self, images, texts, attention_mask=None, token_type_ids=None, labels=None):
        
        image_features, text_features = self.encode(images, texts, attention_mask, token_type_ids)
        
        contrastive_loss = self.contrastive_score(image_features, text_features)
        
        if labels:
            supervised_loss, _ = self.supervised_loss(image_features, text_features, labels)
        
            return self.alpha * contrastive_loss + self.beta * supervised_loss
        else:
            return self.alpha * contrastive_loss





class FedViLT(nn.Module):
    def __init__(self, num_classes):
        super(FedViLT, self).__init__()

        # Initialize the ViLT model
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

        # Classifier
        self.classifier = nn.Linear(self.vilt.config.hidden_size, num_classes)

    def multi_task_loss(image_logits=None, text_logits=None, combined_logits=None, target=None):
        
        image_loss = 0
        text_loss = 0
        combined_loss = 0
        counter = 0
        # Compute the image-only loss
        if image_logits:
            image_loss = F.cross_entropy(image_logits, target)
            counter += 1

        # Compute the text-only loss
        if text_logits:
            text_loss = F.cross_entropy(text_logits, target)
            counter += 1
            
        # Compute the combined image-text loss
        if combined_logits:
            combined_loss = F.cross_entropy(combined_logits, target)
            counter += 1
            
        # Return the sum of the losses
        return (image_loss + text_loss + combined_loss)/counter

    
    def forward(self, image=None, text=None):
        # If both image and text inputs are given
        if image is not None and text is not None:
            outputs = self.vilt(image_input=image, text_input=text)
            combined_logits = self.classifier(outputs.pooler_output)
            return combined_logits
        
        # If only image input is given
        elif image is not None:
            outputs = self.vilt(image_input=image)
            image_logits = self.classifier(outputs.image_output)
            return image_logits

        # If only text input is given
        elif text is not None:
            outputs = self.vilt(text_input=text)
            text_logits = self.classifier(outputs.text_output)
            return text_logits

        return image_logits, text_logits, combined_logits
