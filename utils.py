import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def save_animation(xs, gif_name, interval=300, repeat_delay=5000):
    fig = plt.figure()
    plt.axis('off')
    imgs = []

    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay)
    animate.save(gif_name)



def load_transformed_fashionMNIST(img_size, batch_size, path="data"):
    """
    Downloads and loads Fashion MNIST dataset from given path, default="data".
    Applies transformations like resizing, scaling, etc
    Returns dataloader of specified batch_size.
    """

    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)
    
    train_set = torchvision.datasets.FashionMNIST(path, download=True, train=True, transform=data_transform)
    test_set = torchvision.datasets.FashionMNIST(path, download=True, train=False, transform=data_transform)

    data = torch.utils.data.ConcatDataset([train_set, test_set])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return data, dataloader


def show_tensor_image(image):
    """
    Takes in a tensor image.
    Scales, normalizes, converts to PIL Image and displays using plt.imshow.
    """

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(image[0].detach().cpu()))
    

def to_image(tensor, to_pil=True):
    tensor = (tensor + 1) / 2
    ones = torch.ones_like(tensor)
    tensor = torch.min(torch.stack([tensor, ones]), 0)[0]
    zeros = torch.zeros_like(tensor)
    tensor = torch.max(torch.stack([tensor, zeros]), 0)[0]
    if not to_pil:
        return tensor
    return transforms.functional.to_pil_image(tensor)


def plot_generated_images(noise, result):
    plt.figure(figsize=(8,8))
    nrows = 1
    ncols = 2
    samples = {
        "Noise" : noise,
        "Generated Image" : result
    }
    for i, (title, img) in enumerate(samples.items()):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_title(title)
        show_tensor_image(img)
    plt.show()