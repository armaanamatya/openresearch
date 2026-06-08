"""
Custom optimizer implementations for Adam paper reproduction (arXiv:1412.6980).
from __future__ import annotations added for Python <3.10 union type compat.

All optimizers use param_groups structure (like PyTorch built-ins) so that
learning rate can be updated externally for 1/sqrt(t) scheduling:
    optimizer.param_groups[0]['lr'] = base_lr / math.sqrt(step + 1)
    optimizer.step()
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class BaseOptimizer:
    """Shared scaffolding: param_groups + state dict + zero_grad."""

    def __init__(self, params, defaults: dict):
        params = list(params)
        if len(params) == 0:
            raise ValueError("No parameters to optimize")
        self.param_groups = [{'params': params, **defaults}]
        self.state = {}  # keyed by id(param)
        # Pre-allocate per-parameter state
        self._init_state(params)

    def _init_state(self, params):
        pass  # subclasses override

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self):
        raise NotImplementedError

    @property
    def lr(self):
        return self.param_groups[0]['lr']

    @lr.setter
    def lr(self, value):
        self.param_groups[0]['lr'] = value


# ─── Adam (Algorithm 1) ────────────────────────────────────────────────────────

class AdamOptimizer(BaseOptimizer):
    """Adam optimizer — Algorithm 1 from Kingma & Ba (2014).

    Update rule:
        m_t = β1·m_{t-1} + (1-β1)·g_t            (biased first moment)
        v_t = β2·v_{t-1} + (1-β2)·g_t²            (biased second raw moment)
        m̂_t = m_t / (1-β1^t)                      (bias-corrected first)
        v̂_t = v_t / (1-β2^t)                      (bias-corrected second)
        θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)      (ε outside sqrt)

    Defaults: α=0.001, β1=0.9, β2=0.999, ε=1e-8 (Algorithm 1 caption, Section 2).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bias_correction=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        bias_correction=bias_correction)
        super().__init__(params, defaults)

    def _init_state(self, params):
        for p in params:
            self.state[id(p)] = {
                'm': torch.zeros_like(p.data),  # biased first moment
                'v': torch.zeros_like(p.data),  # biased second raw moment
                't': 0,                          # step counter
            }

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            bias_correction = group['bias_correction']

            for p in group['params']:
                if p.grad is None:
                    continue

                s = self.state[id(p)]
                s['t'] += 1
                t = s['t']

                g = p.grad.data
                if wd != 0.0:
                    g = g + wd * p.data

                # Biased moment estimates
                s['m'] = b1 * s['m'] + (1.0 - b1) * g
                s['v'] = b2 * s['v'] + (1.0 - b2) * (g * g)

                if bias_correction:
                    m_hat = s['m'] / (1.0 - b1 ** t)
                    v_hat = s['v'] / (1.0 - b2 ** t)
                else:
                    # No bias correction — reveals instability at early steps
                    # when β2 is close to 1 (Section 6.4 experiment)
                    m_hat = s['m']
                    v_hat = s['v']

                # θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)   [ε outside sqrt]
                p.data.sub_(lr * m_hat / (v_hat.sqrt() + eps))


# ─── AdaMax (Algorithm 2) ─────────────────────────────────────────────────────

class AdaMaxOptimizer(BaseOptimizer):
    """AdaMax — Algorithm 2 from Kingma & Ba (2014), Section 7.1.

    Infinity-norm variant of Adam:
        m_t = β1·m_{t-1} + (1-β1)·g_t
        u_t = max(β2·u_{t-1}, |g_t|)             (no bias correction on u_t)
        θ_t = θ_{t-1} - (α/(1-β1^t))·m_t / u_t

    Defaults: α=0.002, β1=0.9, β2=0.999 (Section 7.1).
    """

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, params):
        for p in params:
            self.state[id(p)] = {
                'm': torch.zeros_like(p.data),  # biased first moment
                'u': torch.zeros_like(p.data),  # infinity-norm
                't': 0,
            }

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                s = self.state[id(p)]
                s['t'] += 1
                t = s['t']

                g = p.grad.data
                if wd != 0.0:
                    g = g + wd * p.data

                # m_t = β1·m_{t-1} + (1-β1)·g_t
                s['m'] = b1 * s['m'] + (1.0 - b1) * g

                # u_t = max(β2·u_{t-1}, |g_t|)   — no bias correction
                s['u'] = torch.max(b2 * s['u'], g.abs())

                # θ_t = θ_{t-1} - (α/(1-β1^t)) · m_t / u_t
                step_size = lr / (1.0 - b1 ** t)
                p.data.sub_(step_size * s['m'] / (s['u'] + eps))


# ─── SGD with Nesterov Momentum ───────────────────────────────────────────────

class SGDNesterovOptimizer(BaseOptimizer):
    """SGD with Nesterov momentum — baseline from Section 6.

    v_t = μ·v_{t-1} - lr·g_t
    θ_t = θ_{t-1} + μ·v_t - lr·g_t
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, params):
        for p in params:
            self.state[id(p)] = {
                'v': torch.zeros_like(p.data),
            }

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                if wd != 0.0:
                    g = g + wd * p.data

                s = self.state[id(p)]
                v_prev = s['v']
                # Nesterov: update velocity then lookahead
                v_new = mu * v_prev - lr * g
                s['v'] = v_new
                # θ_t = θ_{t-1} + μ·v_t - lr·g_t  (Nesterov form)
                p.data.add_(mu * v_new - lr * g)


# ─── AdaGrad ──────────────────────────────────────────────────────────────────

class AdaGradOptimizer(BaseOptimizer):
    """AdaGrad — baseline from Section 6.

    G_t = G_{t-1} + g_t²
    θ_t = θ_{t-1} - α·g_t / √G_t   (AdaGrad as per Section 6 notation)
    """

    def __init__(self, params, lr=0.01, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, params):
        for p in params:
            self.state[id(p)] = {
                'G': torch.zeros_like(p.data),  # accumulated squared gradient
            }

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                if wd != 0.0:
                    g = g + wd * p.data

                s = self.state[id(p)]
                # G_t = G_{t-1} + g_t²
                s['G'] += g * g
                # θ_t = θ_{t-1} - α·g_t / √(Σg²)
                p.data.sub_(lr * g / (s['G'].sqrt() + eps))


# ─── RMSProp ──────────────────────────────────────────────────────────────────

class RMSPropOptimizer(BaseOptimizer):
    """RMSProp — baseline from Section 6.

    E[g²]_t = ρ·E[g²]_{t-1} + (1-ρ)·g_t²
    θ_t = θ_{t-1} - α·g_t / √(E[g²]_t + ε)
    """

    def __init__(self, params, lr=0.001, rho=0.9, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, params):
        for p in params:
            self.state[id(p)] = {
                'Eg2': torch.zeros_like(p.data),
            }

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                if wd != 0.0:
                    g = g + wd * p.data

                s = self.state[id(p)]
                s['Eg2'] = rho * s['Eg2'] + (1.0 - rho) * (g * g)
                p.data.sub_(lr * g / (s['Eg2'].sqrt() + eps))


# ─── AdaDelta ─────────────────────────────────────────────────────────────────

class AdaDeltaOptimizer(BaseOptimizer):
    """AdaDelta — baseline from Section 6.

    E[g²]_t = ρ·E[g²]_{t-1} + (1-ρ)·g_t²
    Δθ_t = -√(E[Δθ²]_{t-1} + ε) / √(E[g²]_t + ε) · g_t
    E[Δθ²]_t = ρ·E[Δθ²]_{t-1} + (1-ρ)·Δθ_t²
    """

    def __init__(self, params, rho=0.95, eps=1e-6, weight_decay=0.0):
        defaults = dict(rho=rho, eps=eps, weight_decay=weight_decay, lr=1.0)
        super().__init__(params, defaults)

    def _init_state(self, params):
        for p in params:
            self.state[id(p)] = {
                'Eg2':   torch.zeros_like(p.data),
                'Edelta': torch.zeros_like(p.data),
            }

    def step(self):
        for group in self.param_groups:
            rho = group['rho']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                if wd != 0.0:
                    g = g + wd * p.data

                s = self.state[id(p)]
                s['Eg2'] = rho * s['Eg2'] + (1.0 - rho) * (g * g)

                rms_g = (s['Eg2'] + eps).sqrt()
                rms_delta = (s['Edelta'] + eps).sqrt()

                delta = -(rms_delta / rms_g) * g
                s['Edelta'] = rho * s['Edelta'] + (1.0 - rho) * (delta * delta)
                p.data.add_(delta)


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_optimizer(name: str, params, lr: Optional[float] = None, **kwargs):
    """Factory for custom optimizers. Uses paper-default lr when lr=None."""
    name = name.lower()
    if name == 'adam':
        return AdamOptimizer(params, lr=lr or 0.001, **kwargs)
    elif name == 'adamax':
        return AdaMaxOptimizer(params, lr=lr or 0.002, **kwargs)
    elif name in ('sgd', 'sgd_nesterov', 'nesterov'):
        return SGDNesterovOptimizer(params, lr=lr or 0.01, **kwargs)
    elif name == 'adagrad':
        return AdaGradOptimizer(params, lr=lr or 0.01, **kwargs)
    elif name == 'rmsprop':
        return RMSPropOptimizer(params, lr=lr or 0.001, **kwargs)
    elif name == 'adadelta':
        return AdaDeltaOptimizer(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
