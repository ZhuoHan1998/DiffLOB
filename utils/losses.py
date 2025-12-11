import torch
import torch.optim as optim
import numpy as np
from .sde_lib import VESDE, VPSDE, subVPSDE

def get_sde_loss_fn(sde, continuous = True, reduce_mean = True, likelihood_weighting = True, eps = 1e-5):
    """
    Create a loss function for training with arbirary SDEs.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.
    Returns:
      A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, cond):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.
          cond:  Condition

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = get_score_fn(sde, model, continuous = continuous)
        t = torch.rand(batch.shape[0], device = batch.device) * (sde.T - eps) + eps # t in [eps, sde.T], float number
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t, cond)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim = -1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim = -1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_score_fn(sde, model, continuous = False):
    """
    Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
        A score function.
    """

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, t, cond, enable_motion, enable_control):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * (sde.N - 1)
                score = model(x, labels, cond, enable_motion, enable_control)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model(x, labels, cond, enable_motion, enable_control)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t, cond, enable_motion, enable_control):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model(x, labels, cond, enable_motion, enable_control)
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn

def get_smld_loss_fn(vesde, reduce_mean = False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    # smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims = (0,))
    smld_sigma_array = vesde.discrete_sigmas
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, cond):
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device = batch.device) # labels in [0, vesde.N - 1], int number
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model(perturbed_data, labels, cond)
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim = -1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, reduce_mean = True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, cond, enable_motion, enable_control):
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device = batch.device) # labels in [0, vpsde.N - 1], int number
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        score = model(perturbed_data, labels, cond, enable_motion, enable_control)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim = -1)
        loss = torch.mean(losses)
        return loss
    return loss_fn


def get_loss_fn(sde, continuous = False, reduce_mean = True, likelihood_weighting = False):
    """
    If continuous = True, return get_sde_loss_fn; else return get_smld_loss_fn or get_ddpm_loss_fn.
    Note that no matter what sde you choose, as long as continuous = True, it'll always return get_sde_loss_fn.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, 
                                  continuous = True,
                                  reduce_mean = reduce_mean,
                                  likelihood_weighting = likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, reduce_mean = reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, reduce_mean = reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
    
    return loss_fn
