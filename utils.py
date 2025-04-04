import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Variance schedule to sample noise at eavh time step, variables

ncols = 15
nrows = 10
start = 0.0001
end = 0.02
time_steps = ncols * nrows
beta = torch.linspace(start, end, time_steps)
time_steps = len(beta)

# Forward diffusion variables
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)
sqrt_alpha_bar = torch.sqrt(alpha_bar)  # Mean Coefficient
sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)  # St. Dev. Coefficient

# Reverse diffusion variables
sqrt_alpha_inv = torch.sqrt(1 / alpha)
pred_noise_coeff = (1 - alpha) / torch.sqrt(1 - alpha_bar)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def q(x_0, t):
    """
    The forward diffusion process
    Returns the noise applied to an image at timestep t
    x_0: the original image
    t: timestep
    """
    t = t.int()
    noise = torch.randn_like(x_0)
    sqrt_a_bar_t = sqrt_alpha_bar[t, None, None, None]
    sqrt_one_minus_a_bar_t = sqrt_one_minus_alpha_bar[t, None, None, None]

    x_t = sqrt_a_bar_t * x_0 + sqrt_one_minus_a_bar_t * noise
    return x_t, noise

@torch.no_grad()
def reverse_q(x_t, t, e_t):
    """
    The reverse diffusion process
    Returns the an image with the noise from time t removed and time t-1 added.
    model: the model used to remove the noise
    x_t: the noisy image at time t
    t: timestep
    model_args: additional arguments to pass to the model
    """
    t = t.int()
    pred_noise_coeff_t = pred_noise_coeff[t]
    sqrt_a_inv_t = sqrt_alpha_inv[t]
    u_t = sqrt_a_inv_t * (x_t - pred_noise_coeff_t * e_t)
    if t[0] == 0:  # All t values should be the same
        return u_t  # Reverse diffusion complete!
    else:
        beta_t = beta[t - 1]  # Apply noise from the previos timestep
        new_noise = torch.randn_like(x_t)
        return u_t + torch.sqrt(beta_t) * new_noise

def get_loss(model, x_0, t, *model_args):
    x_noisy, noise = q(x_0, t)
    noise_pred = model(x_noisy, t, *model_args)
    return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_images(model, img_ch, img_size, ncols, *model_args, axis_on=False):
    # Noise to generate images from
    x_t = torch.randn((1, img_ch, img_size, img_size), device=device)
    plt.figure(figsize=(8, 8))
    hidden_rows = time_steps / ncols
    plot_number = 1

    # Go from T to 0 removing and adding noise until t = 0
    for i in range(0, time_steps)[::-1]:
        t = torch.full((1,), i, device=device).float()
        e_t = model(x_t, t, *model_args)  # Predicted noise
        x_t = reverse_q(x_t, t, e_t)
        if i % hidden_rows == 0:
            ax = plt.subplot(1, ncols+1, plot_number)
            if not axis_on:
                ax.axis('off')
            show_tensor_image(x_t.detach().cpu())
            plot_number += 1
    plt.show()


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