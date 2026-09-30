"""
MuonAdamW - Hybrid Optimizer combining Muon and AdamW
(based on moonlight Muon and KellerJordan Muon)

Uses Muon (MomentUm Orthogonalized by Newton-schulz) for >=2D parameters,
and AdamW for remaining parameters (1D, embeddings, etc.).

References:
    https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
    https://github.com/KellerJordan/Muon/blob/master/muon.py
"""
import math

import torch

from torch.optim import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Store original dtype to ensure compatibility with mixed precision training
    original_dtype = G.dtype

    # Determine compute dtype:
    # - If input is float32, use bfloat16 for efficient computation
    # - If input is already float16/bfloat16, keep the same dtype
    if G.dtype == torch.float32:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = G.dtype

    X = G.to(compute_dtype)

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X.to(original_dtype)

    return X


class MuonAdamW(Optimizer):
    """
    MuonAdamW - Hybrid optimizer combining Muon and AdamW.

    For >=2D parameters (e.g., weight matrices in linear/conv layers), this optimizer uses
    Muon (MomentUm Orthogonalized by Newton-schulz), which internally runs standard SGD-momentum
    and then performs an orthogonalization post-processing step, replacing each 2D parameter's
    update with the nearest orthogonal matrix via a Newton-Schulz iteration (stably run in bfloat16).

    For remaining parameters (1D params like biases/norms, embeddings, etc.), it falls back to AdamW.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW with weight decay. Any parameters in
        `muon_params` which are {0, 1}-D or are detected as being the embed or lm_head will be
        optimized by AdamW as well.
        adamw_nowd_params: The parameters to be optimized by AdamW without weight decay (1d params
        like norms/biases/LayerScale, and 0d params like logit_scale/logit_bias).
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(self,
                 lr=1e-3,
                 wd=0.1,
                 muon_params=None,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 adamw_params=None,
                 adamw_nowd_params=None,
                 adamw_betas=(0.9, 0.999),
                 adamw_eps=1e-8):
        defaults = dict(lr=lr,
                        wd=wd,
                        momentum=momentum,
                        nesterov=nesterov,
                        ns_steps=ns_steps,
                        adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps)

        muon_params = list(muon_params) if muon_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        adamw_nowd_params = list(
            adamw_nowd_params) if adamw_nowd_params is not None else []

        param_groups = []
        if len(muon_params) > 0 or len(adamw_params) > 0:
            param_groups.append({
                'params': muon_params + adamw_params,
                'wd': wd,
            })
        if len(adamw_nowd_params) > 0:
            param_groups.append({
                'params': adamw_nowd_params,
                'wd': 0.,
            })
        super().__init__(param_groups, defaults)

        for p in muon_params:
            assert p.ndim >= 2, f"Muon requires parameters with ndim >= 2, got {p.ndim}"
            self.state[p]["use_muon"] = True
        for p in adamw_params + adamw_nowd_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio

        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue

                original_shape = g.shape

                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                p.data.add_(u.view(original_shape), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [
                p for p in group["params"] if not self.state[p]["use_muon"]
            ]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                m_hat = buf1 / bias_correction1
                v_hat = buf2 / bias_correction2

                # v_hat <- v_hat.sqrt() + eps
                v_hat.sqrt_().add_(eps)
                # m_hat <- m_hat / (v_hat.sqrt() + eps)
                m_hat.div_(v_hat)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                p.data.add_(m_hat, alpha=-lr)

        return loss


class MuonSGD(Optimizer):
    """
    MuonSGD - Hybrid optimizer combining Muon and SGD.

    For >=2D parameters (e.g., weight matrices in linear/conv layers), this optimizer uses
    Muon (MomentUm Orthogonalized by Newton-schulz), which internally runs standard SGD-momentum
    and then performs an orthogonalization post-processing step, replacing each 2D parameter's
    update with the nearest orthogonal matrix via a Newton-Schulz iteration (stably run in bfloat16).

    For remaining parameters (1D params like biases/norms, embeddings, etc.), it falls back to SGD.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        sgd_params: The parameters to be optimized by SGD with weight decay. Any parameters in
        `muon_params` which are {0, 1}-D or are detected as being the embed or lm_head will be
        optimized by SGD as well.
        sgd_nowd_params: The parameters to be optimized by SGD without weight decay (1d params
        like norms/biases/LayerScale, and 0d params like logit_scale/logit_bias).
        sgd_momentum: The momentum for the internal SGD fallback.
        sgd_nesterov: Whether to use Nesterov-style momentum in the SGD fallback.
    """

    def __init__(self,
                 lr=1e-3,
                 wd=0.1,
                 muon_params=None,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 sgd_params=None,
                 sgd_nowd_params=None,
                 sgd_momentum=0.9,
                 sgd_nesterov=False):
        defaults = dict(lr=lr,
                        wd=wd,
                        momentum=momentum,
                        nesterov=nesterov,
                        ns_steps=ns_steps,
                        sgd_momentum=sgd_momentum,
                        sgd_nesterov=sgd_nesterov)

        muon_params = list(muon_params) if muon_params is not None else []
        sgd_params = list(sgd_params) if sgd_params is not None else []
        sgd_nowd_params = list(
            sgd_nowd_params) if sgd_nowd_params is not None else []

        param_groups = []
        if len(muon_params) > 0 or len(sgd_params) > 0:
            param_groups.append({
                'params': muon_params + sgd_params,
                'wd': wd,
            })
        if len(sgd_nowd_params) > 0:
            param_groups.append({
                'params': sgd_nowd_params,
                'wd': 0.,
            })
        super().__init__(param_groups, defaults)

        for p in muon_params:
            assert p.ndim >= 2, f"Muon requires parameters with ndim >= 2, got {p.ndim}"
            self.state[p]["use_muon"] = True
        for p in sgd_params + sgd_nowd_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio

        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue

                original_shape = g.shape

                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u.view(original_shape), alpha=-adjusted_lr)

            ############################
            #       SGD backup         #
            ############################

            params = [
                p for p in group["params"] if not self.state[p]["use_muon"]
            ]
            lr = group['lr']
            weight_decay = group["wd"]
            sgd_momentum = group["sgd_momentum"]
            sgd_nesterov = group["sgd_nesterov"]

            for p in params:
                g = p.grad
                if g is None:
                    continue

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(sgd_momentum).add_(g)
                if sgd_nesterov:
                    g = g.add(buf, alpha=sgd_momentum)
                else:
                    g = buf

                p.data.add_(g, alpha=-lr)

        return loss
