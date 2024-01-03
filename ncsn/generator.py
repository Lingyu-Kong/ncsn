import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from ncsn.data_loader.data_loader import get_data_loader
from ncsn.models.denoise_score_matching import anneal_dsm_loss
from ncsn.models.sliced_score_matching import anneal_ssm_loss
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

class ScoreBasedGenerator(object):
    """
    Image Generator based on Score-Based Generative model and Annealing Langevin Dynamics.
    Reference: https://arxiv.org/pdf/1907.05600.pdf
    """
    def __init__(
        self,
        scorenet: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        sm_strategy: str = "dsm",
        sigma_begin: float = 1.0,
        sigma_end: float = 0.01,
        num_classes: int = 10,
        num_epochs: int = 500000,
        max_step: int = 200000,
        img_size: int = 32,
        num_channels: int = 3,
    ):
        self.scorenet = scorenet
        self.optimizer = optimizer
        self.device = device    
        self.sm_strategy = sm_strategy
        self.sigmas = torch.from_numpy(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes))).float().to(device)
        self.num_epochs = num_epochs
        self.max_step = max_step
        self.img_size = img_size
        self.num_channels = num_channels
        self.scorenet.to(device)
        
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader = None,
    ):
        if os.path.exists("results"):
            os.system("rm -rf results/*")
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/images", exist_ok=True)
        os.makedirs("results/models", exist_ok=True)
        
        
        self.scorenet.train()
        test_iter = iter(test_loader) if test_loader is not None else None
        step = 0
        pbar = tqdm(total=self.max_step, desc=f"Training")
        test_loss = None 
        for epoch in range(self.num_epochs):
            for x, _ in train_loader:
                step += 1
                x = x.to(self.device)
                x = x / 256. * 255. + torch.rand_like(x) / 256.
                timesteps = torch.randint(0, len(self.sigmas), (x.shape[0],)).to(self.device)
                if self.sm_strategy == "dsm":
                    loss = anneal_dsm_loss(self.scorenet, x, timesteps, self.sigmas)
                elif self.sm_strategy == "ssm":
                    loss = anneal_ssm_loss(self.scorenet, x, timesteps, self.sigmas)
                else:
                    raise NotImplementedError(f"Score Matching Strategy {self.sm_strategy} not implemented.")
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({'Train-Loss': loss.item(), 'Test-Loss': test_loss.item() if test_loss is not None else 0.0})
                pbar.update(1)
                
                if step % 100 == 0 and test_iter is not None:
                    self.scorenet.eval()
                    try:
                        test_X, _ = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, _ = next(test_iter)
                    test_X = test_X.to(self.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    test_timesteps = torch.randint(0, len(self.sigmas), (test_X.shape[0],)).to(self.device)
                    
                    with torch.no_grad():
                        test_loss = anneal_dsm_loss(self.scorenet, test_X, test_timesteps, self.sigmas)
                    self.scorenet.train()
                    
                if step % 10000 == 0:
                    self.sample_image(save_path=f"results/images/{step}.gif")
                    torch.save(self.scorenet.state_dict(), f"results/models/{step}.pth")
                    
                if step >= self.max_step:
                    print(f"Training finished at step {step}.")
                    return

    def anneal_Langevin_Dynamics(
        self,
        x_mod: torch.Tensor,
        n_steps_each_sigma: int = 100,
        step_lr: float = 0.00002,
    ):
        images = []
        with torch.no_grad():
            for t, sigma in enumerate(self.sigmas):
                timesteps = torch.ones(x_mod.shape[0], dtype=torch.long).to(self.device) * t
                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2
                for s in range(n_steps_each_sigma):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2).to(self.device)
                    grad = self.scorenet(x_mod, timesteps)
                    x_mod = x_mod + step_size * grad + noise
        return images
                
    def sample_image(self, num_rows: int = 4, save_path: str = None):
        save_path = save_path.split(".")[0]
        self.scorenet.eval()
        num_samples = num_rows ** 2
        x_mod = torch.rand(num_samples, self.num_channels, self.img_size, self.img_size).to(self.device)
        images = self.anneal_Langevin_Dynamics(x_mod)
        img_movie = []
        for i, sample in enumerate(images):
            sample = sample.view(num_samples, self.num_channels, self.img_size, self.img_size)
            image_grid = make_grid(sample, nrow=num_rows).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            if i % 10 == 0:
                im = Image.fromarray(image_grid)
                img_movie.append(im)
            if i == len(images) - 1:
                im.save(save_path + ".png")
        img_movie[0].save(save_path + ".gif", save_all=True, append_images=img_movie[1:], duration=1, loop=0)
        
                
            

        
                    
                    
            