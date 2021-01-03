from torchvision import models
import torch

dir(models)

from torchvision import transforms

donusturmek = transforms.Compose([          # Giriş görseli Üzerinde yapılacak değişimler

 transforms.Resize(256),                    # Görüntüyü 256x256 boyutuna dönüştürmek

 transforms.CenterCrop(224),                # Görüntüyü merkez etrafında 224 × 224 piksel olacak şekilde kırpmak.

 transforms.ToTensor(),                     # Görüntüyü PyTorch Tensor veri türüne dönüştürmek.

 transforms.Normalize(                      # Ortalama ve standart sapmasını  -

 mean=[0.485, 0.456, 0.406],                # belirtilen değerlere ayarlayarak -

 std=[0.229, 0.224, 0.225]                  # görüntüyü normalleştirmek

 )])
 
 
 
from PIL import Image
resim = Image.open("kopek.jpg")
resim


resim_t = donusturmek(resim)
yigin_t = torch.unsqueeze(resim_t, 0)


alexnet = models.alexnet(pretrained=True)


print(alexnet)


cikti = alexnet(yigin_t)
print(cikti.shape)

with open('imagenet_classes.txt') as k:

  siniflar = [line.strip() for line in k.readlines()]
  
  
_, endeksler = torch.sort(cikti, descending=True)
yuzde = torch.nn.functional.softmax(cikti, dim=1)[0] * 100
[(siniflar[indeks], yuzde[indeks].item()) for indeks in endeksler[0][:7]]
