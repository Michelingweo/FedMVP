import os
import math
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from torch import nn
# import spacya
# spacy.load('en_core_web_sm')
from spacy.lang.en import English
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from utils.options import args_parser


args = args_parser()

DATASETS_PATH = '/home/lfc5481/project/MCFL/data/'

# CUB_200_2011
CUB_200_2011_PATH = os.path.join(DATASETS_PATH, 'cub')
# Captions and metadata
CUB_200_2011_METADATA_PATH           = os.path.join(CUB_200_2011_PATH, 'Metadata_dummy.pth')
CUB_200_2011_IMG_ID_TO_CAPS_PATH     = os.path.join(CUB_200_2011_PATH, 'cub_200_2011_img_id_to_caps.pth')
# All images
CUB_200_2011_IMGS_64_PATH            = os.path.join(CUB_200_2011_PATH, 'imgs_64x64.pth')
CUB_200_2011_IMGS_128_PATH           = os.path.join(CUB_200_2011_PATH, 'imgs_128x128.pth')
CUB_200_2011_IMGS_256_PATH           = os.path.join(CUB_200_2011_PATH, 'imgs_256x256.pth')
# Training/Validation split images
CUB_200_2011_TRAIN_VAL_IMGS_64_PATH  = os.path.join(CUB_200_2011_PATH, 'imgs_train_val_64x64.pth')
CUB_200_2011_TRAIN_VAL_IMGS_128_PATH = os.path.join(CUB_200_2011_PATH, 'imgs_train_val_128x128.pth')
CUB_200_2011_TRAIN_VAL_IMGS_256_PATH = os.path.join(CUB_200_2011_PATH, 'imgs_train_256x256.pth')
# Testing split images
CUB_200_2011_TEST_IMGS_64_PATH       = os.path.join(CUB_200_2011_PATH, 'imgs_test_64x64.pth')
CUB_200_2011_TEST_IMGS_128_PATH      = os.path.join(CUB_200_2011_PATH, 'imgs_test_128x128.pth')
CUB_200_2011_TEST_IMGS_256_PATH      = os.path.join(CUB_200_2011_PATH, 'imgs_test_256x256.pth')
# GLOVE word embeddings for CUB 200 2011
CUB_200_2011_GLOVE_PATH = os.path.join(CUB_200_2011_PATH, 'glove_relevant_embeddings.pth')
CUB_200_2011_D_VOCAB = 1750
# CUB generated image and text
CUB_200_2011_GENERATED_IMGS_PATH = os.path.join(CUB_200_2011_PATH, 'cub_generated_image_all.pth')
CUB_200_2011_GENERATED_TXTS_PATH = os.path.join(CUB_200_2011_PATH, 'cub_generated_text_all.pth')

# OXFORD_FLOWERS_102
OXFORD_FLOWERS_102_PATH = os.path.join(DATASETS_PATH, 'flower_data')
OXFORD_FLOWERS_102_IMG_ID_TO_CAPS_PATH     = os.path.join(OXFORD_FLOWERS_102_PATH, 'oxford_flowers_102_img_id_to_caps.pth')
# Captions and metadata
OXFORD_FLOWERS_102_METADATA_PATH           = os.path.join(OXFORD_FLOWERS_102_PATH, 'metadata.pth')
# All images
OXFORD_FLOWERS_102_IMGS_64_PATH            = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_64x64.pth')
OXFORD_FLOWERS_102_IMGS_128_PATH           = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_128x128.pth')
OXFORD_FLOWERS_102_IMGS_256_PATH           = os.path.join(OXFORD_FLOWERS_102_PATH, 'flower_imgs_ALL_256x256.pth')
# Training/Validation split images
OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_64_PATH  = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_train_val_64x64.pth')
OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_128_PATH = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_train_val_128x128.pth')
OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_256_PATH = os.path.join(OXFORD_FLOWERS_102_PATH, 'flower_imgs_trainval_256x256.pth')
# Testing split images
OXFORD_FLOWERS_102_TEST_IMGS_64_PATH       = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_test_64x64.pth')
OXFORD_FLOWERS_102_TEST_IMGS_128_PATH      = os.path.join(OXFORD_FLOWERS_102_PATH, 'imgs_test_128x128.pth')
OXFORD_FLOWERS_102_TEST_IMGS_256_PATH      = os.path.join(OXFORD_FLOWERS_102_PATH, 'flower_imgs_test_256x256.pth')

#Generated images and texts
OXFORD_FLOWERS_102_GENERATED_IMGS_PATH = os.path.join(OXFORD_FLOWERS_102_PATH, 'flower_generated_image_all.pth')
OXFORD_FLOWERS_102_GENERATED_TXTS_PATH = os.path.join(OXFORD_FLOWERS_102_PATH, 'flower_generated_text.pth')



# GLOVE 300 DIM EMBEDDINGS OF 5102 WORDS (from scratchgan paper)
GLOVE_5102_300_PATH = os.path.join(DATASETS_PATH, 'glove-300-dimensional-embeddings-for-5102-words', 'glove_5102_300.pth')

# CUB 200 2011, OXFORD FLOWERS 102, EMNLP 2017 AND WIKITEXT 103 COMBINED VOCAB
COMBINED_WORD_FREQ_PATH = os.path.join(DATASETS_PATH, 'cub_oxford_emnlp_wikitext_word_freqs.pth')
WORD_ID_TO_WORD_5K_PATH = os.path.join(DATASETS_PATH, 'combined-vocab', 'word_id_to_word_5k.pth')
WORD_ID_TO_WORD_10K_PATH = os.path.join(DATASETS_PATH, 'combined-vocab', 'word_id_to_word_10k.pth')



class CUB_200_2011(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, split='all', d_image_size=256, transform=None, should_pad=False, pad_to_length=86, no_start_end=False, text_return_type='bert', **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        assert d_image_size in (64, 128, 256)
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.d_image_size = d_image_size
        self.transform = transform
        self.should_pad = should_pad
        self.pad_to_length = pad_to_length # maximum cap length = 86
        self.no_start_end = no_start_end
        # bert: bert tokenizer(bs,256), encoded: original encoded captions(bs, 256), para: bert tokenizer para (bs, 1000), text: strings
        self.text_return_type = text_return_type
        # if args.model == 'fedbert' or 'fedmvp':
        #     self.text_return_type = 'bert'


        metadata = torch.load(CUB_200_2011_METADATA_PATH)

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']
        self.class_id_to_class_name = metadata['class_id_to_class_name']
        self.class_name_to_class_id = metadata['class_name_to_class_id']

        #dummy sentence
        self.dummy_encode = metadata['dummy']['dummy_bertencode']
        self.dummy_mask = metadata['dummy']['dummy_bertmask']
        self.dummy_tokenid = metadata['dummy']['dummy_berttokenid']
        
        # captions
        self.img_id_to_text = metadata['img_id_to_text']
        self.img_id_to_bertencode = metadata['img_id_to_bertencode']
        self.img_id_to_bertmask = metadata['img_id_to_bertmask']
        self.img_id_to_bertrokenid = metadata['img_id_to_berttokenid']
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.word_id_to_word = metadata['word_id_to_word']
        self.word_to_word_id = metadata['word_to_word_id']
        self.pad_token     = self.word_to_word_id['<PAD>']
        self.start_token   = self.word_to_word_id['<START>']
        self.end_token     = self.word_to_word_id['<END>']
        self.unknown_token = self.word_to_word_id['<UNKNOWN>']

        self.d_vocab = metadata['num_words']
        self.num_captions_per_image = metadata['num_captions_per_image']

        nlp = English()
        # nlp = spacy.load("en_core_web_sm")
        # self.tokenizer = nlp.Defaults.create_tokenizer(nlp) # Create a Tokenizer with the default settings for English including punctuation rules and exceptions
        self.tokenizer = nlp.tokenizer # Create a Tokenizer with the default settings for English including punctuation rules and exceptions

        # images
        self.img_missing_ids = []
        self.text_missing_ids = []
        if split == 'all':
            self.img_ids = metadata['img_ids']
            imgs_path = CUB_200_2011_IMGS_256_PATH
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
            # print(f'trainset length===={len(self.img_ids)}')

            imgs_path = CUB_200_2011_TRAIN_VAL_IMGS_256_PATH
            
            #modality missing
            if args.missing_ratio > 0:
                
                print(f'===={100 * args.missing_ratio}% data missed=====')
                
                missing_num = int(args.missing_ratio * len(self.img_ids))
                img_missing_num = int(args.img_missing_ratio * missing_num)
                text_missing_num = missing_num - img_missing_num
                
                self.img_missing_ids = random.sample(self.img_ids, img_missing_num)
                remaining_ids = [n for n in self.img_ids if n not in self.img_missing_ids]
                self.text_missing_ids = random.sample(remaining_ids, text_missing_num)
            
        else:
            self.img_ids = metadata['test_img_ids']
            # print(f'testset length===={len(self.img_ids)}')
            imgs_path = CUB_200_2011_TEST_IMGS_256_PATH

        self.imgs = torch.load(imgs_path)
        # assert self.imgs.size()[1:] == (3, d_image_size, d_image_size) and self.imgs.dtype == torch.uint8
        
        if args.missing_ratio > 0:
            self.generated_imgs = torch.load(CUB_200_2011_GENERATED_IMGS_PATH)
            self.generated_txts = torch.load(CUB_200_2011_GENERATED_TXTS_PATH)
            self.generated_bertencode = self.generated_txts['img_id_to_bertencode']
            self.generated_bertmask = self.generated_txts['img_id_to_bertmask']
            self.generated_berttokenid = self.generated_txts['img_id_to_berttokenid']
        
        

    def encode_caption(self, cap):
        words = [token.text for token in self.tokenizer(cap)]
        return [self.word_to_word_id.get(word, self.unknown_token) for word in words]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2 # 2 since we need a start token and an end token.
        cap = cap[:max_len] # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length] # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        
        img_id = self.img_ids[idx]
        if self.img_missing_ids and args.model == 'fedmvp' and img_id in self.img_missing_ids:
            img = self.generated_imgs[img_id]
        else:
            img = self.imgs[img_id]
        
        if self.transform:
            img = self.transform(img)
        
        class_id = self.img_id_to_class_id[img_id]
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        texts = self.img_id_to_text[img_id]
        
        if self.text_missing_ids and args.model == 'fedmvp' and img_id in self.text_missing_ids:
            bertencode_cap = self.generated_bertencode[img_id]
            bertmask = self.generated_bertmask[img_id]
            berttokenid = self.generated_berttokenid[img_id]
        else:
            
            bertencode_caps = self.img_id_to_bertencode[img_id]
            bertmasks = self.img_id_to_bertmask[img_id]
            berttokenids = self.img_id_to_bertrokenid[img_id]

            cap_idx = torch.randint(low=0, high=self.num_captions_per_image, size=(1,)).item()
            # print(f'cap_idx:{cap_idx}')
            encoded_cap = encoded_caps[cap_idx]

            text = texts[cap_idx]
            bertencode_cap = bertencode_caps[cap_idx]
            bertmask = bertmasks[cap_idx]
            berttokenid = berttokenids[cap_idx]
            
            # if self.should_pad:
            #     if self.no_start_end:
            #         encoded_cap, cap_len = self.pad_without_start_end(encoded_cap)
            #     else:
            #         encoded_cap, cap_len = self.pad_caption(encoded_cap)
            #     return img, class_id, torch.as_tensor(encoded_cap, dtype=torch.int)
         
        if self.text_return_type == 'text':
            return img, class_id, text, idx, img_id
        elif self.text_return_type == 'bert':
            return img, class_id, (bertencode_cap[0:196], bertmask[0:196], berttokenid[0:196])
            # return img, class_id, (torch.as_tensor(self.dummy_encode[0:196],dtype=torch.long), 
            #                        torch.as_tensor(self.dummy_mask[0:196],dtype=torch.long), 
            #                        torch.as_tensor(self.dummy_tokenid[0:196],dtype=torch.long))
        
        else:
            return img, class_id, torch.as_tensor(bert_encoded, dtype=torch.int)




class OxfordFlowers102(Dataset):
    """If should_pad is True, need to also provide a pad_to_length. Padding also adds <START> and <END> tokens to captions."""
    def __init__(self, split='all', d_image_size=256, transform=None, should_pad=True, pad_to_length=256, no_start_end=False, text_return_type='bert', **kwargs):
        super().__init__()

        assert split in ('all', 'train_val', 'test')
        assert d_image_size in (64, 128, 256)
        if should_pad:
            assert pad_to_length >= 3 # <START> foo <END> need at least length 3.

        self.split = split
        self.d_image_size = d_image_size
        self.transform = transform
        self.should_pad = should_pad
        self.pad_to_length = pad_to_length
        self.no_start_end = no_start_end
        self.text_return_type = text_return_type

        metadata = torch.load(OXFORD_FLOWERS_102_METADATA_PATH)

        # labels
        self.img_id_to_class_id = metadata['img_id_to_class_id']

        # self.class_id_to_class_name = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
        self.class_id_to_class_name = metadata['class_id_to_class_name']
        self.class_name_to_class_id = {c: i for i, c in enumerate(self.class_id_to_class_name)}
        
        # captions
        self.img_id_to_encoded_caps = metadata['img_id_to_encoded_caps']
        self.img_id_to_bertencode = metadata['img_id_to_bertencode']
        self.img_id_to_bertmask = metadata['img_id_to_bertmask']
        self.img_id_to_bertrokenid = metadata['img_id_to_berttokenid']

        self.word_id_to_word = metadata['word_id_to_word']
        self.word_to_word_id = metadata['word_to_word_id']
        self.pad_token     = 0
        self.start_token   = 1747
        self.end_token     = 1748
        self.unknown_token = 1749

        self.d_vocab = metadata['num_words']
        self.num_captions_per_image = metadata['num_captions_per_image']

        nlp = English()
        self.tokenizer = nlp.tokenizer # Create a Tokenizer with the default settings for English including punctuation rules and exceptions

        # images
        self.img_missing_ids = []
        self.text_missing_ids = []
        if split == 'all':
            self.img_ids = metadata['img_ids']
            if d_image_size == 64:
                imgs_path = OXFORD_FLOWERS_102_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = OXFORD_FLOWERS_102_IMGS_128_PATH
            else:
                imgs_path = OXFORD_FLOWERS_102_IMGS_256_PATH
        elif split == 'train_val':
            self.img_ids = metadata['train_val_img_ids']
            
            #modality missing
            if args.missing_ratio > 0:
                
                print(f'===={100 * args.missing_ratio}% data missed=====')

                
                missing_num = int(args.missing_ratio * len(self.img_ids))
                img_missing_num = int(args.img_missing_ratio * missing_num)
                text_missing_num = missing_num - img_missing_num
                
                self.img_missing_ids = random.sample(self.img_ids, img_missing_num)
                remaining_ids = [n for n in self.img_ids if n not in self.img_missing_ids]
                self.text_missing_ids = random.sample(remaining_ids, text_missing_num)
            
            
            if d_image_size == 64:
                imgs_path = OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_128_PATH
            else:
                imgs_path = OXFORD_FLOWERS_102_TRAIN_VAL_IMGS_256_PATH
        else:
            self.img_ids = metadata['test_img_ids']
            if d_image_size == 64:
                imgs_path = OXFORD_FLOWERS_102_TEST_IMGS_64_PATH
            elif d_image_size == 128:
                imgs_path = OXFORD_FLOWERS_102_TEST_IMGS_128_PATH
            else:
                imgs_path = OXFORD_FLOWERS_102_TEST_IMGS_256_PATH

        self.imgs = torch.load(imgs_path)
        
        if args.missing_ratio > 0:
            self.generated_imgs = torch.load(OXFORD_FLOWERS_102_GENERATED_IMGS_PATH)
            self.generated_txts = torch.load(OXFORD_FLOWERS_102_GENERATED_TXTS_PATH)
            self.generated_bertencode = self.generated_txts['img_id_to_bertencode']
            self.generated_bertmask = self.generated_txts['img_id_to_bertmask']
            self.generated_berttokenid = self.generated_txts['img_id_to_berttokenid']
        # assert self.imgs.size()[1:] == (3, d_image_size, d_image_size) and self.imgs.dtype == torch.uint8

    def encode_caption(self, cap):
        words = [token.text for token in self.tokenizer(cap)]
        return [self.word_to_word_id.get(word, self.unknown_token) for word in words]

    def decode_caption(self, cap):
        if isinstance(cap, torch.Tensor):
            cap = cap.tolist()
        return ' '.join([self.word_id_to_word[word_id] for word_id in cap])

    def pad_caption(self, cap):
        max_len = self.pad_to_length - 2 # 2 since we need a start token and an end token.
        cap = cap[:max_len] # truncate to maximum length.
        cap = [self.start_token] + cap + [self.end_token]
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap = list(cap)
        padding = list(padding)

        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def pad_without_start_end(self, cap):
        cap = cap[:self.pad_to_length] # truncate to maximum length.
        cap_len = len(cap)
        padding = [self.pad_token]*(self.pad_to_length - cap_len)
        cap += padding
        return torch.tensor(cap, dtype=torch.long), cap_len

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        
        
        if self.img_missing_ids and args.model == 'fedmvp' and img_id in self.img_missing_ids:
            img = self.generated_imgs[img_id]
        else:
            img = self.imgs[img_id]
            
        if self.transform:
            img = self.transform(img)
        class_id = int(self.img_id_to_class_id[img_id]) - 1
        
        encoded_caps = self.img_id_to_encoded_caps[img_id]
        
        if self.text_missing_ids and args.model == 'fedmvp' and img_id in self.text_missing_ids:
            bertencode_cap = self.generated_bertencode[img_id]
            bertmask = self.generated_bertmask[img_id]
            berttokenid = self.generated_berttokenid[img_id]
            # print(f'bertencode:{bertencode_caps.shape}')
            # print(f'bertmasks:{bertmasks.shape}')
            # print(f'berttokenids:{berttokenids.shape}')
            
        else:
            bertencode_caps = self.img_id_to_bertencode[img_id]
            bertmasks = self.img_id_to_bertmask[img_id]
            berttokenids = self.img_id_to_bertrokenid[img_id]

            cap_idx = torch.randint(low=0, high=self.num_captions_per_image, size=(1,)).item()

            encoded_cap = encoded_caps[cap_idx]
            bertencode_cap = bertencode_caps[cap_idx]
            bertmask = bertmasks[cap_idx]
            berttokenid = berttokenids[cap_idx]

        if self.text_return_type == 'encode':
            if self.should_pad:
                if self.no_start_end:
                    encoded_cap, cap_len = self.pad_without_start_end(encoded_cap)
                else:
                    encoded_cap, cap_len = self.pad_caption(encoded_cap)
                return img, class_id, encoded_cap
            return img, class_id, encoded_cap
        elif self.text_return_type == 'bert':
            return img, class_id, (bertencode_cap[0:196], bertmask[0:196], berttokenid[0:196])
        else:
            return img, class_id, bertencode_cap



data_transforms = {
            'train': transforms.Compose([

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 均值，方差
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }



def get_cub_200_2011(split='test',args=args,d_batch=4, should_pad=True, shuffle=True, **kwargs):

    if split == 'test':
        # d_batch = args.bs
        d_batch = args.bs
        transform = transforms.Compose([

            transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.),
            # transforms.CenterCrop(224),
            transforms.Resize(size=(224,224)),
        ])
    else:
        d_batch = args.local_bs
        transform = transforms.Compose([
            transforms.RandomRotation(45),  # 随机旋转，-45到45度
            transforms.CenterCrop(224),  # 从中心开始裁剪，将其剪为224
            transforms.RandomHorizontalFlip(p=0.5),  # 有p的概率随机水平反转
            transforms.RandomVerticalFlip(p=0.5),  # 垂直反转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  ##亮度，对比度，饱和度，色相
            transforms.RandomGrayscale(p=0.025),  
            transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.),
            # transforms.CenterCrop(224),
            transforms.Resize(size=(224,224)),
        ])


    train_set = CUB_200_2011(split=split, transform=transform, should_pad=should_pad, **kwargs)


    if not should_pad:
        def collate_fn(samples):
            imgs, class_ids, caps = zip(*samples)
            imgs = torch.stack(imgs)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
            return imgs, class_ids, caps
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=collate_fn) # num_workers linux:4, windows:0
    
    else:
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, num_workers=2, pin_memory=True)
    return train_set, train_loader



def get_oxford_flowers_102(split='train_val',args=args, transform=None, d_batch=4, should_pad=True, shuffle=True, **kwargs):

    if split == 'test':
        d_batch = args.bs
        transform = transforms.Compose([

            transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.),
            # transforms.CenterCrop(224),
            transforms.Resize(size=(224, 224)),
        ])
    else:
        d_batch = args.local_bs
        transform = transforms.Compose([
            transforms.RandomRotation(45),  # random rotation, -45 to +45
            transforms.CenterCrop(224),  # crop from the center, size=224
            transforms.RandomHorizontalFlip(p=0.5),  # 50% flip
            transforms.RandomVerticalFlip(p=0.5),  
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), 
            transforms.RandomGrayscale(p=0.025), 
            transforms.Lambda(lambda x: (x.float() / 255.) * 2. - 1.),
            # transforms.CenterCrop(224),
            transforms.Resize(size=(224, 224)),
        ])

    train_set = OxfordFlowers102(split=split, transform=transform, should_pad=should_pad, **kwargs)

    if not should_pad:
        def collate_fn(samples):
            imgs, class_ids, caps = zip(*samples)
            imgs = torch.stack(imgs)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
            return imgs, class_ids, caps
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=d_batch, shuffle=shuffle, num_workers=2, pin_memory=True)
    return train_set, train_loader


def to_one_hot(labels, d_classes):
    """
    :param labels: integer tensor of shape (d_batch, *)
    :param d_classes : number of classes
    :return: float tensor of shape (d_batch, *, d_classes), one hot representation of the labels
    """
    return torch.zeros(*labels.size(), d_classes, device=labels.device).scatter_(-1, labels.unsqueeze(-1), 1)


###################################### TESTS ######################################

def test_get_cub_200_2011():
    train_set, train_loader = get_cub_200_2011(should_pad=True, pad_to_length=25)
    print('Size of vocabulary:', train_set.d_vocab)
    print('Number of images:', len(train_set))
    print('Number of batches:', len(train_loader))
    batch = next(iter(train_loader))
#     imgs, caps = batch
#     print(imgs.size(), imgs.dtype, imgs.max(), imgs.min())
#     print(caps)
    imgs, class_ids, caps, cap_lens = batch
    print(imgs.size(), imgs.dtype, imgs.min(), imgs.max())
    print(class_ids.size(), class_ids.dtype, class_ids)
    print(caps.size(), caps.dtype, caps)
    print(cap_lens.size(), cap_lens.dtype, cap_lens)
    for cap in caps:
        print(train_set.decode_caption(cap))

    # visualize
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    imgs = torch.stack([denorm(img) for img in imgs]).permute(0, 2, 3, 1).cpu()
    plt.figure(figsize=(1*8,4*8))
    for i, img in enumerate(imgs):
        plt.subplot(4, 1, i+1)
        plt.imshow(img)
        plt.axis('off')
        cap = train_set.decode_caption(caps[i])
        plt.title(cap)
    plt.suptitle('some images from the dataset')

def test_oxford_flowers_102():
    train_set, train_loader = get_oxford_flowers_102(should_pad=True, pad_to_length=25)
    print('Size of vocabulary:', train_set.d_vocab)
    print('Number of images:', len(train_set))
    print('Number of batches:', len(train_loader))
    batch = next(iter(train_loader))
#     imgs, caps = batch
#     print(imgs.size(), imgs.dtype, imgs.max(), imgs.min())
#     print(caps)
    imgs, class_ids, caps, cap_lens = batch
    print(imgs.size(), imgs.dtype, imgs.min(), imgs.max())
    print(class_ids.size(), class_ids.dtype, class_ids)
    print(caps.size(), caps.dtype, caps)
    print(cap_lens.size(), cap_lens.dtype, cap_lens)
    for cap in caps:
        print(train_set.decode_caption(cap))

    # visualize
    denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
    imgs = torch.stack([denorm(img) for img in imgs]).permute(0, 2, 3, 1).cpu()
    plt.figure(figsize=(1*8,4*8))
    for i, img in enumerate(imgs):
        plt.subplot(4, 1, i+1)
        plt.imshow(img)
        plt.axis('off')
        cap = train_set.decode_caption(caps[i])
        plt.title(cap)
    plt.suptitle('some images from the dataset')


def mkdir_p(path):
    """Make directory and all necessary parent directories given by the path."""
    os.makedirs(path, exist_ok=True)

def get_timestamp():
    """Return the current date and time in year, month, day, hour, minute and second format."""
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%M_%S')

def uniform_in_range(low=0., high=1., size=(1,), **kwargs):
    return (high - low) * torch.rand(*size, **kwargs) + low

def get_biased_coin(probability_of_heads=0.5):
    """Get a biased coin that returns True on heads."""
    return lambda: torch.rand(1).item() < probability_of_heads

def get_cub_200_2011_glove_embeddings(fine_tune=False):
    GLOVE_D_EMBED = 50
    glove_embeddings = torch.load(CUB_200_2011_GLOVE_PATH)
    assert glove_embeddings.size() == (CUB_200_2011_D_VOCAB, GLOVE_D_EMBED)
    assert glove_embeddings.dtype == torch.float
    embed = nn.Embedding(CUB_200_2011_D_VOCAB, GLOVE_D_EMBED)
    embed.weight == nn.Parameter(glove_embeddings)
    embed.weight.requires_grad = fine_tune

    return embed

def text_embeds_to_encoded_texts(text_embeds, ref_embeds):
    """
    Find closest words using euclidean distance.
    Args:
        text_embeds (Tensor): float tensor of shape (d_batch, d_max_seq_len, d_text_embed)
        ref_embeds (Tensor): float tensor of shape (d_vocab, d_text_embed)
    Returns:
        encoded_texts list(list(int)): list of list of word indices.
    """
    encoded_texts = []
    for text_embed in text_embeds:
        encoded_text = []
        for word_embed in text_embed:
            errs = (ref_embeds - word_embed.unsqueeze(0))
            dists = (errs**2).sum(dim=1)
            word_id = dists.argmin().item()
            encoded_text.append(word_id)
        encoded_texts.append(encoded_text)
    return encoded_texts


