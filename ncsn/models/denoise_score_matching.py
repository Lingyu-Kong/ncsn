import torch
import torch.nn as nn

def anneal_dsm_loss(scorenet, samples, labels, sigmas, anneal_power=2.):
    """
    Follow the equation (5) and (6) in https://arxiv.org/pdf/1907.05600.pdf
    """
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

# def anneal_dsm_loss(
#     scorenet: nn.Module,
#     samples: torch.Tensor,
#     timesteps: torch.Tensor,
#     sigmas: torch.Tensor,
#     anneal_power: float=2.,    
# ):
#     """
#     Follow the equation (5) and (6) in https://arxiv.org/pdf/1907.05600.pdf
#     """
#     used_sigmas = sigmas[timesteps].view(samples.shape[0], *([1] * len(samples.shape[1:])))
#     perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
#     target = -1/(used_sigmas**2) * (perturbed_samples - samples)
#     scores = scorenet(perturbed_samples, timesteps)
#     target = target.view(target.shape[0], -1)
#     scores = scores.view(scores.shape[0], -1)
#     loss = 1/2. * torch.sum((scores - target)**2, dim=-1) * used_sigmas.squeeze()**anneal_power
#     return loss.mean(dim=0)
    
    