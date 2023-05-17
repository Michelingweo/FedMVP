import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib.image as mpimg
import numpy as np
from PIL import Image




dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

images = dalle2(
    ['a butterfly trying to escape a tornado'],
    cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
)

# save your image (in this example, of size 256x256)
# print(images.dtype())
img = images.to(torch.device("cpu"))
img = img[0]
save_image(img,'Dalle2.png')
torch.save(img,'Dalle2.pt')


transform = T.ToPILImage()
img = transform(img)
img.show()

# plt.imshow(img) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()
#
# # save
# # 适用于保存任何 matplotlib 画出的图像，相当于一个 screencapture
# plt.savefig('DALLE2.png')