import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net

image_nc=1
epochs = 60
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = '/cephfs/dongyulong/work/wudao/model/0.0003_model.pk'
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5]),
])

mnist_dataset = torchvision.datasets.MNIST(
            root='/cephfs/dongyulong/work/wudao/data/', # 数据存放的路径
            transform=transform,
            train=True,
            download=True
)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

advGAN = AdvGAN_Attack(device,
                       targeted_model,
                        model_num_labels,
                        image_nc,
                        BOX_MIN,
                        BOX_MAX
)

advGAN.train(dataloader, epochs)
