# Data

Use CUB 200 and Oxford flower as the datasets for multi-modal fine-grained categorization.

****
### CUB 200
The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is the most widely-used dataset for fine-grained visual categorization task. It contains 11,788 images of 200 subcategories belonging to birds, 5,994 for training and 5,794 for testing. Each image has detailed annotations: 1 subcategory label, 15 part locations, 312 binary attributes and 1 bounding box. The textual information comes from Reed et al.. They expand the CUB-200-2011 dataset by collecting fine-grained natural language descriptions. Ten single-sentence descriptions are collected for each image. The natural language descriptions are collected through the Amazon Mechanical Turk (AMT) platform, and are required at least 10 words, without any information of subcategories and actions.

**Num Classes:** 200

**Modality:** image, text

**Document Descript:**

Please download the image modality from http://www.vision.caltech.edu/datasets/cub_200_2011/

****
### Oxford Flower 102
Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The flowers chosen to be flower commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images.

The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.
**Num Classes:** 102

**Modality:** image, text

**Document Descript:**
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
****


_metadata:_ .pth file that contains all the necessary info, both CUB and Flower datasets share the same metadata structure.

| key                      | dtype | value                                                                                                                                 |
|--------------------------|-------|---------------------------------------------------------------------------------------------------------------------------------------|
| 'img_ids'                | list  | all the image ids                                                                                                                     |
| 'img_id_to_class_id'     | dict  | image id to class                                                                                                                     |
| 'class_id_to_class_name' | dict  |                                                                                                                                       |
| 'class_name_to_class_id' | dict  |                                                                                                                                       |
| 'train_val_class_ids'    |       |                                                                                                                                       |
| 'test_class_ids'         |       |                                                                                                                                       |
| 'train_val_img_ids'      | list  |                                                                                                                                       |
| 'test_img_ids'           | list  |                                                                                                                                       |
| 'word_freq'              | dict  |                                                                                                                                       |
| 'word_id_to_word'        | dict  |                                                                                                                                       |
| 'word_to_word_id'        | dict  |                                                                                                                                       |
| 'img_id_to_encoded_caps' | dict  | id:[[encoded caption]*10] (encoded based on word freq)                                                                                |
| 'num_imgs'               | int   | CUB:11788                                                                                                                             |
| 'num_classes'            | int   | CUB:200, Flower:102                                                                                                                   |
| 'num_captions_per_image' | int   | 10                                                                                                                                    |
| 'num_words'              | int   |                                                                                                                                       |
| 'img_id_to_text'         | dict  | id:[[caption str]...]                                                                                                                 |
| 'img_id_to_bertsent'     | dict  | id:[[encoded caption]*10] (by pretrained BERT tokenizer https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#berttokenizer) |
| 'img_id_to_bertpara'     | dict  | id:[encoded caption] (concatenated 10 sentences encoded by pretrained BERT tokenizer)                                                 |












