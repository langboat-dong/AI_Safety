import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net

image_nc=1
batch_size = 1

gen_input_nc = image_nc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the pretrained model
pretrained_model = '/cephfs/dongyulong/work/wudao/model/0.0003_model.pk'
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_60.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5]),
])

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST(
            root='/cephfs/dongyulong/work/wudao/data/', 
            transform=transform,
            train=False,
            download=True
)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)

save_nums = 0

for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    print(adv_img.shape)
    if pred_lab != test_label:
        if save_nums < 10:
            transforms.ToPILImage()(adv_img.squeeze(0)).save(f'images/advgan_{save_nums}.jpg')
            save_nums += 1
        else:
            break

