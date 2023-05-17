
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import vit_b_16, resnet18
import math
import os
from transformers import BertTokenizer, BertModel, BertConfig
import timm

MODEL_SAVE_PATH = './save/trained_model/'
VIT_CUB_PATH = os.path.join(MODEL_SAVE_PATH, 'vit_cub.pth')
VIT_FLOWER_PATH = os.path.join(MODEL_SAVE_PATH, 'vit_flower.pth')
BERT_CUB_PATH = os.path.join(MODEL_SAVE_PATH, 'bert_cub.pth')
BERT_FLOWER_PATH = os.path.join(MODEL_SAVE_PATH, 'bert_flower.pth')

from utils.options import args_parser
from models.transformer_networks import *
from models.Nets import *





args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')



def l2_distance(emb1, emb2):

    diff = emb1 - emb2
    l2_distance = torch.sqrt(torch.pow(diff, 2).sum())
    return l2_distance



class FedMVP(nn.Module):
    '''
    Args:
          args: experiment parameters
          common_dim = dimension while passing through the encoders
          latent_dim = embedding dimension (input of the classifier)
          loss_type: all:supervised+ multimodal contrastive + marginal
                    mc: supervsied + multimodal contrastive
                    marginal: supervised + marginal
                    supervised: only supervised loss
    '''

    def __init__(self, args, common_dim=256, latent_dim=512, loss_type="all"):
        super(FedMVP, self).__init__()

        self.args = args

        self.common_dim = common_dim # hidden states embedding dimension
        self.latent_dim = latent_dim # output latent embedding dimension
        self.loss_type = loss_type # supervised (only sup), mc(mc+sup), marginal(marg+sup), all
        self.alpha = 0.01 # hyper para for MC loss
        self.beta = 0.5 # hyper para for marginal


        if self.args.pretrain:
            if self.args.dataset == 'cub':
                vit_para = torch.load(VIT_CUB_PATH)
                bert_para = torch.load(BERT_CUB_PATH)
            else:
                vit_para = torch.load(VIT_FLOWER_PATH)
                bert_para = torch.load(BERT_FLOWER_PATH)
                
            self.image_processor = vit_b_16(pretrained=False)
            self.image_processor.load_state_dict(vit_para)
            
            
            self.image_processor.heads.head = Linear(in_features=self.image_processor.heads.head.in_features, out_features=latent_dim)
            
            self.text_processor = BertModel.from_pretrained('bert-base-uncased')
            self.text_processor.load_state_dict(bert_para)
            for p in self.parameters():
                p.requires_grad = False
            
            self.text_mapping = nn.Linear(768,latent_dim)
            self.text_input_mapping = nn.Linear(768,common_dim)

        else:
            # self.text_embedding = nn.Embedding(num_embeddings=8418, embedding_dim=256)
            # self.text_processor = GRUEncoder(input_dim=common_dim, hidden_dim=common_dim, latent_dim=latent_dim, timestep=common_dim, batch_first=True)
            self.text_processor = GRUEncoder(args=args, vocab_size=30522, embedding_dim=256, hidden_size=256,
                          num_classes=256)
            
            self.image_processor = timm.create_model('cspresnet50', pretrained=self.args.pretrain, num_classes=latent_dim)


        # self.joint_processor = JointEncoder(args=self.args,hidden_size=common_dim,output_size=latent_dim)
        self.joint_processor = Transformer_JointEncoder(embed_dim=256, latent_dim=512, patch_size=16, num_heads=4, num_layers=3)


        self.encoder = CommonEncoder(common_dim=latent_dim, latent_dim=latent_dim)


        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, self.args.num_classes)
        # self.classifier = MLP(dim_in=latent_dim,dim_hidden1=256, dim_hidden2=256, dim_out=self.args.num_classes)
        # self.criterion = nn.CrossEntropyLoss()


    def encode(self, x, sample=False):

        # If we have complete observations
        if None not in x:

            joint_representation = self.encoder(self.processors[-1](x))
            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
            output += joint_representation

            return self.classifier(output)

        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)

            # Forward classifier
            output = self.proj2(F.dropout(F.relu(self.proj1(latent)), p=0.4, training=self.training))
            output += latent

            return self.classifier(output)

    def forward(self, image, text, attention_mask=None, token_id=None):
        # Forward pass through the modality specific encoders
        batch_representations = []

        if self.args.pretrain:
         
            img_ = self.image_processor(image)
            text_output = self.text_processor(text, attention_mask=attention_mask, token_type_ids=token_id)
            text_ = text_output.pooler_output
            text_input = self.text_input_mapping(text_output.last_hidden_state)

            text_specific = self.encoder(self.text_mapping(text_))
            image_specific = self.encoder(img_)
            
            # ==========================
            # text_specific = self.encoder(img_)
            # image_specific = self.encoder(self.text_mapping(text_))
            
            # ===========================

        else:
            image_specific = self.encoder(self.image_processor(image))
            text_specific = self.encoder(self.text_processor(text))
            
            # ==========================
            # text_specific = self.encoder(self.image_processor(image))
            # image_specific = self.encoder(self.text_processor(text))
            
            # ===========================
            

        batch_representations.append(image_specific)
        batch_representations.append(text_specific)
        # print(f'image:{image.shape}')
        # print(f'text input:{text_input.shape}')
        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.joint_processor(image, text_input))
        # print(f'joint rep shape:{joint_representation.shape}')
        joint_representation = joint_representation.mean(dim=0)
        # print(f'MEAN joint rep shape:{joint_representation.shape}')
        batch_representations.append(joint_representation)

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.1, training=self.training))

        output += joint_representation
        output = self.classifier(output)
        return output, batch_representations

    def mc_loss(self, prediction, target, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0

        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair

            #concatnate the two tensors along dim 0
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            ) # [2*Batch size, latent_dim]

            # torch.mm : matrix multiplication
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            ) # [2*B, 2*B]
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            
            
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()

            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        return self.alpha * joint_mod_loss_sum.mean()

    def supervised_loss(self, prediction, target):
        return F.cross_entropy(prediction, target)


    def marginal_loss(self, joint_pred, target, joint_loss, batch_representations):
        image_representation = batch_representations[0]
        text_representation = batch_representations[1]
        joint_representation = batch_representations[2]

        image_pred = self.classifier(image_representation)
        text_pred = self.classifier(text_representation)

        image_loss = F.cross_entropy(image_pred, target)
        text_loss = F.cross_entropy(text_pred, target)

        marginal_i = torch.tensor(0.)
        marginal_t = torch.tensor(0.)

        if image_loss < joint_loss:
            marginal_i = l2_distance(joint_representation, image_representation)
            # marginal_i = self.criterion(joint_pred, image_pred)

        if text_loss < joint_loss:
            marginal_t = l2_distance(joint_representation, text_representation)
            # marginal_t = self.criterion(joint_pred, text_pred)

        marginal_loss = ( marginal_i + marginal_t) / self.args.local_bs

        return self.beta * marginal_loss




    def training_loss(self, output, batch_representations, target, temperature=0.1):

        batch_size = target.shape[0]

        if self.loss_type == 'mc':
            mc_loss = self.mc_loss(prediction=output, target=target, batch_representations=batch_representations, temperature=temperature, batch_size=batch_size)
            supervised_loss = self.supervised_loss(prediction=output, target=target)

            loss = supervised_loss + mc_loss
        elif self.loss_type == 'marginal':
            supervised_loss = self.supervised_loss(prediction=output, target=target)
            marginal_loss = self.marginal_loss(joint_pred=output, target=target, joint_loss=supervised_loss,
                                               batch_representations=batch_representations)

            loss = supervised_loss + marginal_loss
        elif self.loss_type == 'supervised':
            supervised_loss = self.supervised_loss(prediction=output, target=target)
            loss = supervised_loss

        else:
            mc_loss = self.mc_loss(prediction=output, target=target, batch_representations=batch_representations,
                                   temperature=temperature, batch_size=batch_size)
            supervised_loss = self.supervised_loss(prediction=output, target=target)
            marginal_loss = self.marginal_loss(joint_pred=output, target=target, joint_loss=supervised_loss,
                                               batch_representations=batch_representations)

            # print(f'mc_loss:{mc_loss} == {mc_loss.shape}')
            # print(f'marginal_loss:{marginal_loss}=={marginal_loss.shape}')
            # print(f'supervised loss:{supervised_loss}=={supervised_loss.shape}')

            loss = supervised_loss + mc_loss + marginal_loss

        return loss

    def validation_step(self, image, text, target, temperature=0.1):

        batch_size = image.shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(image, text)

        # Compute loss
        if self.loss_type == 'mc':
            mc_loss = self.mc_loss(prediction=output, target=target, batch_representations=batch_representations,
                                   temperature=temperature, batch_size=batch_size)
            supervised_loss = self.supervised_loss(prediction=output, target=target)

            loss = supervised_loss + mc_loss
        elif self.loss_type == 'marginal':
            supervised_loss = self.supervised_loss(prediction=output, target=target)
            marginal_loss = self.marginal_loss(joint_pred=output, target=target, joint_loss=supervised_loss,
                                               batch_representations=batch_representations)

            loss = supervised_loss + marginal_loss
        elif self.loss_type == 'supervised':
            supervised_loss = self.supervised_loss(prediction=output, target=target)
            loss = supervised_loss

        else:
            mc_loss = self.mc_loss(prediction=output, target=target, batch_representations=batch_representations,
                                   temperature=temperature, batch_size=batch_size)
            supervised_loss = self.supervised_loss(prediction=output, target=target)
            marginal_loss = self.marginal_loss(joint_pred=output, target=target, joint_loss=supervised_loss,
                                               batch_representations=batch_representations)

            loss = supervised_loss + mc_loss + marginal_loss

        return loss, output



class CommonEncoder(nn.Module):

    def __init__(self, common_dim, latent_dim):
        super(CommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)




class GRUEncoder(nn.Module):
    def __init__(self, args, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=2, dropout=0.1):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.args = args

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.args.device)
        output, h_n = self.gru(x, h0)
        # print(f'output shape:{output.shape}')
        # print(output[0].shape)
        # print(output.mean(dim=1).shape)
        # print(f'h_n shape:{h_n.shape}')
        # x = self.dropout(torch.cat((h_n[:, -1, :self.hidden_size], h_n[:, 0, self.hidden_size:]), dim=1))
        x = self.dropout(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1))
        # print(x.shape)
        # x = self.fc(x)
        return x



# class GRUEncoder(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, timestep, batch_first=False):
#         super(GRUEncoder, self).__init__()
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
#
#         self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
#                           batch_first=batch_first)
#         self.projector = nn.Linear(self.hidden_dim*timestep, latent_dim)
#
#         self.ts = timestep
#
#     def forward(self, x):
#         batch = len(x)
#         input = x.reshape(batch, self.ts, self.input_dim).transpose(0, 1)
#         output = self.gru(input)[0].transpose(0, 1)
#         # print(output.shape)
#         return self.projector(output.flatten(start_dim=1))



# class JointEncoder(nn.Module):
#     def __init__(self, args, hidden_size,  output_size):
#         super(JointEncoder, self).__init__()
#         self.args = args
#         # self.image_encoder = PatchModule(patch_size=16, hidden_size=hidden_size)
#         # self.text_encoder = TextEncoder(input_size)

#         # image patching embedding and text linear embedding
#         self.image_projection = nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=16, stride=16)
#         self.text_projection = nn.Embedding(num_embeddings=5597, embedding_dim=256, padding_idx=0)

#         self.img_to_text_attention = CrossModalityAttention(hidden_size=hidden_size, modality_relation='vt')
#         self.text_to_img_attention = CrossModalityAttention(hidden_size=hidden_size, modality_relation='tv')
#         # self.self_attention = MultiheadAttention(hidden_size, num_heads=1)
#         self.TFlayer = TransformerEncoderLayer(hidden_size*2,
#                                                 num_heads=4,
#                                                 attn_dropout=0.0,
#                                                 relu_dropout=0.0,
#                                                 res_dropout=0.0,
#                                                 attn_mask=False)
#         self.fc = nn.Linear(2 * hidden_size, output_size)

#     def forward(self, image, text):
#         # cub: image [bs, 3, 256, 256]
#         # encode the image and text
#         if self.args.pretrain:
#             image_patch= self.image_projection(image)
#             text = text
#         else:
#             image_patch = self.image_projection(image)
#             text = self.text_projection(text)


#         # apply the crossmodal attention module
#         image_embedding = self.text_to_img_attention(image=image_patch, text=text)
#         text_embedding = self.img_to_text_attention(image=image_patch, text=text)

#         # concatenate the image and text embeddings: 256 -> 512
#         embedding = torch.cat((image_embedding, text_embedding), dim=-1)

#         # apply the multi-head attention module
#         embedding = self.TFlayer(embedding)

#         # apply the fully connected layer
#         embedding = self.fc(embedding)

#         return embedding



class PatchModule(nn.Module):
    def __init__(self, patch_size, hidden_size, output_size):
        super(PatchModule, self).__init__()

        self.patch_size = patch_size
        # self.num_blocks = num_blocks
        # self.num_heads = num_heads
        self.hidden_size = hidden_size
        # self.output_size = output_size

        # define the patch embedding and transformer blocks
        self.patch_embedding = nn.Linear(3*patch_size*patch_size, hidden_size)
        # self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(num_blocks)])
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # divide the image into patches
        patches = self.divide_into_patches(x)

        # apply the patch embedding
        patches = self.patch_embedding(patches)

        # # apply the transformer blocks
        # for transformer_block in self.transformer_blocks:
        #   patches = transformer_block(patches)
        #
        # # apply the fully connected layer
        # patches = self.fc(patches)

        return patches

    def divide_into_patches(self, x):
        # extract the batch size and number of channels from the input tensor
        batch_size, channels, height, width = x.size()

        # compute the number of patches in the vertical and horizontal directions
        num_patches_vertical = height // self.patch_size
        num_patches_horizontal = width // self.patch_size

        # reshape the input tensor into a tensor of patches
        patches = x.view(batch_size, channels, num_patches_vertical, self.patch_size, num_patches_horizontal, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(batch_size, num_patches_vertical*num_patches_horizontal, channels*self.patch_size*self.patch_size)

        return patches



class PatchEmbedding(nn.Module):
    def __init__(self, dim_model: int, patch_size: int, in_channels: int):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, dim_model, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)
        x= x.permute(1, 0, 2)

        return x



class Transformer_JointEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        num_layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, latent_dim, num_heads, num_layers, patch_size, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, layers_type='transformer'):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.embed_scale = 1/math.sqrt(self.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dim)
        self.latent_scale = 1/math.sqrt(self.latent_dim)
        self.latent_positions = SinusoidalPositionalEmbedding(self.latent_dim)

        # self.patch_embedding = SPT(dim=embed_dim, patch_size=patch_size, channels=3)
        self.patch_embedding = PatchEmbedding(dim_model=embed_dim, patch_size=patch_size, in_channels=3)
        
        # self.patch_embedding = OverlappedPatchEmbedding(image_size=224, patch_size=patch_size, stride=8, embedding_dim=embed_dim)
        
        self.attn_mask = attn_mask



        self.image_attention = TransformerEncoderLayer(embed_dim,
                                                       num_heads=num_heads,
                                                       attn_dropout=attn_dropout,
                                                       relu_dropout=relu_dropout,
                                                       res_dropout=res_dropout,
                                                       attn_mask=attn_mask)

        self.img_to_text_attention = TransformerEncoderLayer(embed_dim,
                                                             num_heads=num_heads,
                                                             attn_dropout=attn_dropout,
                                                             relu_dropout=relu_dropout,
                                                             res_dropout=res_dropout,
                                                             attn_mask=attn_mask)
        self.text_to_img_attention = TransformerEncoderLayer(embed_dim,
                                                             num_heads=num_heads,
                                                             attn_dropout=attn_dropout,
                                                             relu_dropout=relu_dropout,
                                                             res_dropout=res_dropout,
                                                             attn_mask=attn_mask)


        self.layers_type = layers_type # transformer or linear

        if self.layers_type == 'transformer':
            self.layers = nn.ModuleList([])
            for layer in range(num_layers):
                new_layer = TransformerEncoderLayer(latent_dim,
                                                    num_heads=num_heads * 2,
                                                    attn_dropout=attn_dropout,
                                                    relu_dropout=relu_dropout,
                                                    res_dropout=res_dropout,
                                                    attn_mask=attn_mask)
                self.layers.append(new_layer)
        else:
            self.layers = nn.Sequential(
                nn.Linear(self.latent_dim, 2*self.latent_dim),
                nn.Linear(2*self.latent_dim, self.latent_dim),
            )

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(self.latent_dim)

    def forward(self, image, text):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        image_patch = self.patch_embedding(image)

        image_embedding = image_patch.transpose(0, 1) # (batch, seq_len, embed_dim) --> (seq_len, batch, embed_dim)
        text_embedding = text.transpose(0, 1)

        image_embedding = self.embed_scale * image_embedding
        text_embedding = self.embed_scale * text_embedding

        image_embedding += self.embed_positions(image_embedding.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        text_embedding += self.embed_positions(text_embedding.transpose(0, 1)[:, :, 0]).transpose(0, 1)


        image_embedding = self.image_attention(x=image_embedding, x_k=image_embedding, x_v=image_embedding)
        
        
        #=====================
        # text_embedding = image_embedding # image only
        # image_embedding =text_embedding # text only
        #=====================
        

        embedding_VT = self.img_to_text_attention(x=image_embedding, x_k=text_embedding, x_v=text_embedding)
        embedding_TV = self.text_to_img_attention(x=text_embedding, x_k=image_embedding, x_v=image_embedding)

        # print(f'embedding_VT:{embedding_VT.shape}')
        # print(f'embedding_TV:{embedding_TV.shape}')

        embedding = torch.cat((embedding_VT, embedding_TV), dim=-1)


        x_in = embedding
        x_in_k = embedding
        x_in_v = embedding

        # embed tokens and positions
        x = self.latent_scale * x_in
        if self.latent_positions is not None:
            x += self.latent_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)



        # x = embedding
        # x_k = embedding
        # x_v = embedding
        # encoder layers
        intermediates = [x]

        if self.layers_type == 'transformer':
            if x_in_k is not None and x_in_v is not None:
                # embed tokens and positions
                x_k = self.latent_scale * x_in_k
                x_v = self.latent_scale * x_in_v
                if self.latent_positions is not None:
                    x_k += self.latent_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0,
                                                                                            1)  # Add positional embedding
                    x_v += self.latent_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0,
                                                                                            1)  # Add positional embedding
                x_k = F.dropout(x_k, p=self.dropout, training=self.training)
                x_v = F.dropout(x_v, p=self.dropout, training=self.training)

            for layer in self.layers:
                if x_in_k is not None and x_in_v is not None:
                    x = layer(x, x_k, x_v)
                else:

                    x = layer(x)
                intermediates.append(x)
        else:
            # print(f'x_shape:{x.shape}')
            x = x.mean(dim=0)
            x = self.layers(x)

        if self.normalize:
            x = self.layer_norm(x)
        # print(f'x_shape:{x.shape}')

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())



class central_jointencoder(nn.Module):
    def __init__(self, args):
        super(central_jointencoder, self).__init__()
        self.args = args
        self.text_processor = BertModel.from_pretrained('bert-base-uncased')
        for p in self.text_processor.parameters():
            p.requires_grad = False
        self.text_mapping = nn.Linear(768, 256)

        self.joint_encoder = Transformer_JointEncoder(embed_dim=256, latent_dim=512, patch_size=16, num_heads=4, num_layers=3)

        self.classifier = MLP(256*2, 256*4, 256, self.args.num_classes)

    def forward(self, image, text, attention_mask=None, token_id=None):


        # text_embedding = self.text_processor(text, attention_mask=attention_mask, token_type_ids=token_id).pooler_output
        text_embedding = self.text_processor(text, attention_mask=attention_mask, token_type_ids=token_id).last_hidden_state
        

        text_embedding = self.text_mapping(text_embedding)
        # print(text_embedding.shape)
        embedding = self.joint_encoder(image, text_embedding) #(196,64,512)
        embedding = embedding.mean(dim=0)
      
        output = self.classifier(embedding)

        return output