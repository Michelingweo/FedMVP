from models.Nets import *
from models.transformer_networks import *
from models.Nets import MLP
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from utils.options import args_parser
from torchvision.models import vit_b_16, resnet18
import math



args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')



class MCFed(nn.Module):
    def __init__(self, args, common_dim=256, latent_dim=512, loss_type="CE"):
        super(MCFed, self).__init__()

        self.args = args

        self.common_dim = common_dim # hidden states embedding dimension
        self.latent_dim = latent_dim # output latent embedding dimension
        self.loss_type = loss_type
        self.alpha = 0.1
        #
        # if self.loss_type == 'CE':
        #     self.text_processor = None
        #     self.image_processor = None
        # else:
        #     self.text_processor = GRUEncoder(input_dim=common_dim, hidden_dim=common_dim, latent_dim=latent_dim,
        #                                      timestep=common_dim)
        #     # self.text_processor = nn.Linear(in_features=common_dim,out_features=latent_dim)
        #     self.image_processor = resnet18(pretrained=True)
        #     self.image_processor.fc = torch.nn.Linear(in_features=self.image_processor.fc.in_features,
        #                                               out_features=latent_dim)


        if self.args.pretrain:
            self.text_processor =GRUEncoder(input_dim=common_dim, hidden_dim=common_dim, latent_dim=latent_dim, timestep=common_dim)
            # self.text_processor = nn.Linear(in_features=common_dim,out_features=latent_dim)
            self.image_processor = resnet18(pretrained=True)
            self.image_processor.fc = torch.nn.Linear(in_features=self.image_processor.fc.in_features, out_features=latent_dim)
        else:
            self.text_processor = GRUEncoder(input_dim=common_dim, hidden_dim=common_dim, latent_dim=latent_dim, timestep=common_dim)


            self.image_processor = resnet18(pretrained=False)
            self.image_processor.fc = torch.nn.Linear(in_features=self.image_processor.fc.in_features,
                                                      out_features=latent_dim)


        # self.joint_processor = JointEncoder(args=self.args,hidden_size=common_dim,output_size=latent_dim)
        self.joint_processor = Transformer_JointEncoder(embed_dim=256, latent_dim=512, patch_size=16, num_heads=4, layers=2)


        self.encoder = AffectEncoder(common_dim=latent_dim, latent_dim=latent_dim)
        # self.encoder = AffectEncoder(common_dim=latent_dim, latent_dim=latent_dim)

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, self.args.num_classes)
        # self.classifier = MLP(dim_in=latent_dim,dim_hidden1=256, dim_hidden2=256, dim_out=self.args.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward_(self, image, text):
        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.joint_processor(image, text))
        joint_representation = joint_representation.mean(dim=0)
        # print(f'joint_representation:{joint_representation.shape}')
        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))

        output += joint_representation
        # print(f'forward output:{output.shape}')
        return self.classifier(output)
    def training_step_(self, image, text, target):
        batch_size = image.shape[0]

        # Forward pass through the encoders
        output = self.forward_(image, text)
        # print(f'output:{output.shape}')
        # Compute contrastive + supervised loss
        loss = self.criterion(output, target)

        return loss, output


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

    def forward(self, image, text):
        # Forward pass through the modality specific encoders
        batch_representations = []

        image_specific = self.encoder(self.image_processor(image))
        batch_representations.append(image_specific)

        text_specific = self.encoder(self.text_processor(text))
        batch_representations.append(text_specific)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.joint_processor(image, text))
        joint_representation = joint_representation.mean(dim=0)
        batch_representations.append(joint_representation)

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))

        output += joint_representation

        return self.classifier(output), batch_representations

    def gmc_loss(self, prediction, target, batch_representations, temperature, batch_size):
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


        supervised_loss = self.criterion(prediction, target)

        loss = torch.mean(self.alpha * joint_mod_loss_sum + supervised_loss)
        # loss = torch.mean(supervised_loss)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict


    def training_step(self, image, text, target, temperature=0.1):

        batch_size = image.shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(image, text)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.gmc_loss(prediction=output, target=target, batch_representations=batch_representations, temperature=temperature, batch_size=batch_size)

        return loss, tqdm_dict

    def validation_step(self, image, text, target, temperature=0.1):

        batch_size = image.shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(image, text)
        # Compute contrastive loss
        loss, tqdm_dict = self.gmc_loss(output, target, batch_representations, temperature, batch_size)
        return loss, tqdm_dict, output




# Affect
class AffectGMC(MCFed):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", scenario='mosei'):
        super(AffectGMC, self).__init__(name, common_dim, latent_dim, loss_type)

        if scenario == 'mosei':
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=74, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=35, hidden_dim=30, latent_dim=latent_dim, timestep=50)
        else:
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=5, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=20, hidden_dim=30, latent_dim=latent_dim, timestep=50)

        self.joint_processor = AffectJointProcessor(latent_dim, scenario)

        self.processors = [
            self.language_processor,
            self.audio_processor,
            self.vision_processor,
            self.joint_processor]

        self.loss_type = loss_type

        self.encoder = AffectEncoder(common_dim=common_dim, latent_dim=latent_dim)

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)



class AffectEncoder(nn.Module):

    def __init__(self, common_dim, latent_dim):
        super(AffectEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.encode = nn.Linear(common_dim, latent_dim)

    def forward(self, x):
        return F.normalize(self.encode(x), dim=-1)




# class GRU(nn.Module):
#     '''
#     input_size: the size of the encoded sentences (CUB, 374)
#     hidden_size: the size of the hidden state in the GRU
#     num_layers: the number of GRU layers
#     output_size: the size of the output embedding (CUB, 512).
#     '''
#     def __init__(self, input_size=374, hidden_size=512, num_layers=2, output_size=512):
#
#         super(GRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # self.text_projection = nn.Embedding(num_embeddings=5597, embedding_dim=256, padding_idx=0)
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # initialize the hidden state with zeros
#         # x = self.text_projection(x)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         h0 = h0.to(args.device)
#
#         # perform the forward pass of the GRU
#         out, _ = self.gru(x, h0)
#
#         # retrieve the last output of the GRU
#         out = out[:, -1, :]
#
#         # apply the fully connected layer
#         out = self.fc(out)
#
#         return out


class GRUEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, timestep, batch_first=False):
        super(GRUEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=batch_first)
        self.projector = nn.Linear(self.hidden_dim*timestep, latent_dim)

        self.ts = timestep

    def forward(self, x):
        batch = len(x)
        input = x.reshape(batch, self.ts, self.input_dim).transpose(0, 1)
        output = self.gru(input)[0].transpose(0, 1)
        # print(output.shape)
        return self.projector(output.flatten(start_dim=1))



class JointEncoder(nn.Module):
    def __init__(self, args, hidden_size,  output_size):
        super(JointEncoder, self).__init__()
        self.args = args
        # self.image_encoder = PatchModule(patch_size=16, hidden_size=hidden_size)
        # self.text_encoder = TextEncoder(input_size)

        # image patching embedding and text linear embedding
        self.image_projection = nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=16, stride=16)
        self.text_projection = nn.Embedding(num_embeddings=5597, embedding_dim=256, padding_idx=0)

        self.img_to_text_attention = CrossModalityAttention(hidden_size=hidden_size, modality_relation='vt')
        self.text_to_img_attention = CrossModalityAttention(hidden_size=hidden_size, modality_relation='tv')
        # self.self_attention = MultiheadAttention(hidden_size, num_heads=1)
        self.TFlayer = TransformerEncoderLayer(hidden_size*2,
                                                num_heads=4,
                                                attn_dropout=0.0,
                                                relu_dropout=0.0,
                                                res_dropout=0.0,
                                                attn_mask=False)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, image, text):
        # cub: image [bs, 3, 256, 256]
        # encode the image and text
        if self.args.pretrain:
            image_patch= self.image_projection(image)
            text = text
        else:
            image_patch = self.image_projection(image)
            text = self.text_projection(text)


        # apply the crossmodal attention module
        image_embedding = self.text_to_img_attention(image=image_patch, text=text)
        text_embedding = self.img_to_text_attention(image=image_patch, text=text)

        # concatenate the image and text embeddings: 256 -> 512
        embedding = torch.cat((image_embedding, text_embedding), dim=-1)

        # apply the multi-head attention module
        embedding = self.TFlayer(embedding)

        # apply the fully connected layer
        embedding = self.fc(embedding)

        return embedding



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


class Transformer_JointEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, latent_dim, num_heads, layers, patch_size, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.embed_scale = 1/math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.latent_scale = math.sqrt(latent_dim)
        self.latent_positions = SinusoidalPositionalEmbedding(latent_dim)

        self.patch_embedding = SPT(dim=embed_dim, patch_size=patch_size, channels=3)
        self.attn_mask = attn_mask

        self.combine_way = 'cat'  # cat or sum

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

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(latent_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(latent_dim)

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

        image_embedding = image_patch.transpose(0, 1)
        text_embedding = text.transpose(0, 1)

        image_embedding = self.embed_scale * image_embedding
        text_embedding = self.embed_scale * image_embedding

        image_embedding += self.embed_positions(image_embedding.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        text_embedding += self.embed_positions(text_embedding.transpose(0, 1)[:, :, 0]).transpose(0, 1)

        embedding_VT = self.img_to_text_attention(x=image_embedding, x_k=text_embedding, x_v=text_embedding)
        embedding_TV = self.text_to_img_attention(x=text_embedding, x_k=image_embedding, x_v=image_embedding)

        # print(f'embedding_VT:{embedding_VT.shape}')
        # print(f'embedding_TV:{embedding_TV.shape}')

        if self.combine_way == 'cat':
            embedding = torch.cat((embedding_VT, embedding_TV), dim=-1)
        else:
            embedding = embedding_TV + embedding_TV
        # print(f'embedding:{embedding.shape}')

        x_in = embedding
        x_in_k = embedding
        x_in_v = embedding

        # embed tokens and positions
        x = self.latent_scale * x_in
        if self.latent_positions is not None:
            x += self.latent_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

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

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())
