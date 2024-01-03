import torch
import torch.nn as nn
import torch.autograd as autograd

def anneal_ssm_loss(
    scorenet: nn.Module,
    samples: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    n_particles: int=1,
):
    """
    Follow equation (3) with annealing strategy in https://arxiv.org/pdf/1907.05600.pdf
    """
    used_sigmas = sigmas[timesteps].view(samples.shape[0], 1, 1, 1)
    used_sigmas = used_sigmas.expand_as(samples).to(samples.device)
    perturbed_samples = samples + used_sigmas * torch.randn_like(samples)
    dup_samples = perturbed_samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_timestep = timesteps.unsqueeze(0).expand(n_particles, *timesteps.shape).contiguous().view(-1)
    dup_samples.requires_grad_(True)
    
    vectors = torch.randn_like(dup_samples)
    
    grad1 = scorenet(dup_samples, dup_timestep)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    grad1 = grad1.view(dup_samples.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = (loss1 + loss2) * (used_sigmas.squeeze() ** 2)
    
    return loss.mean(dim=0)
    
    