"""Optional finite-state CSS diffusion, independent of replacement-policy training.

Q[i,j] = P(next=j | current=i). All supported declaration slots diffuse
independently; owner identity, property, unit and priority remain fixed.
"""
import re

import torch
from torch import nn

CONTRACT = 'css-value-d3pm-v3-size-margin'


def grid(field, unit, bins):
    """Global grids, never centered on an unavailable ground-truth value."""
    from .tree_edits import FIELDS
    if field not in FIELDS:raise ValueError('Unsupported size/margin diffusion field')
    if bins < 3 or bins % 2 != 1:
        raise ValueError('bins must be odd and >=3')
    if unit not in ('px', '%'):
        raise ValueError('Only px and % are diffused; other declarations stay fixed')
    extent = 2048. if unit == 'px' else 200.
    return torch.linspace(0, extent, bins, dtype=torch.float64) if field in ('width', 'height') else torch.linspace(-extent/2, extent/2, bins, dtype=torch.float64)


def numeric_slots(state, fields, bins, minimums=None):
    slots, labels, errors = [], [], []
    for owner, props in enumerate(state[1:], 1):
        for prop, (value, priority) in enumerate(props):
            match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))(px|%)?', value)
            if not match:
                continue
            number = float(match[1]); unit = match[2] or 'px'
            if match[2] is None and number != 0:
                continue
            values = grid(fields[prop], unit, bins)
            minimum=(minimums or {}).get(owner,{}).get(fields[prop])
            if minimum is not None:
                if unit!='px':raise ValueError('Explicit diffusion sizes must be px')
                if minimum>=float(values[-1]):raise ValueError('Border minimum exceeds diffusion grid')
                values=torch.linspace(minimum,float(values[-1]),bins,dtype=torch.float64)
            if not values[0] <= number <= values[-1]:
                continue  # No clipping of out-of-range original declarations.
            index = int((values-number).abs().argmin())
            slots.append({'owner': owner, 'property': prop, 'unit': unit, 'priority': priority})
            if minimum is not None:slots[-1]['minimum']=minimum
            labels.append(index)
            errors.append({'unit': unit, 'absolute_error': abs(float(values[index])-number)})
    return slots, labels, errors


def slot_edit(slot, index, fields, bins):
    values=grid(fields[slot['property']], slot['unit'], bins)
    if 'minimum' in slot:values=torch.linspace(slot['minimum'],float(values[-1]),bins,dtype=torch.float64)
    value = float(values[int(index)])
    return [slot['owner'], fields[slot['property']], f'{value:.8g}{slot["unit"]}', slot['priority']]


class Kernel(nn.Module):
    """Lazy Gaussian random walk plus a tiny uniform component for full support.

    Exact finite-chain posteriors; intentionally not LayoutDiffusion's schedule.
    t runs from 1..T, cumulative[0] is the identity.
    """
    def __init__(self, bins=129, steps=200, sigma_min=.35, sigma_max=2., move_rate=.2):
        super().__init__()
        if bins < 3 or steps < 1 or not 0 < sigma_min <= sigma_max or not 0 < move_rate <= 1:
            raise ValueError('Invalid diffusion kernel')
        self.bins, self.steps = bins, steps
        distance = torch.arange(bins, dtype=torch.float64)
        distance = (distance[:, None]-distance[None, :]).square()
        identity = torch.eye(bins, dtype=torch.float64)
        transitions, cumulative = [identity], [identity]
        for sigma in torch.linspace(sigma_min, sigma_max, steps):
            local = torch.softmax(-distance/(2*float(sigma)**2), dim=-1)
            q = (1-move_rate)*identity + move_rate*((1-1e-6)*local + 1e-6/bins)
            transitions.append(q)
            cumulative.append(cumulative[-1] @ q)
        self.register_buffer('transitions', torch.stack(transitions))
        self.register_buffer('cumulative', torch.stack(cumulative))

    def check_t(self, t):
        if not 1 <= t <= self.steps:
            raise ValueError('t must be in 1..diffusion_steps')

    def sample(self, x0, t, generator=None):
        self.check_t(t)
        return torch.multinomial(self.cumulative[t][x0], 1, generator=generator).squeeze(-1)

    def posterior(self, x0, xt, t):
        self.check_t(t)
        likelihood = self.transitions[t][:, xt].T
        probabilities = self.cumulative[t-1][x0] * likelihood
        return probabilities / probabilities.sum(-1, keepdim=True)

    def reverse(self, logits, xt, t):
        """sum_i p_theta(x0=i) q(x_{t-1} | xt, x0=i), not q(xt|E[x0])."""
        self.check_t(t)
        p0 = logits.double().softmax(-1)
        denominator = self.cumulative[t][:, xt].T
        prior = (p0 / denominator) @ self.cumulative[t-1]
        probabilities = prior * self.transitions[t][:, xt].T
        return probabilities / probabilities.sum(-1, keepdim=True)

    def loss(self, logits, x0, xt, t, auxiliary=.01):
        target = self.posterior(x0, xt, t)
        prediction = self.reverse(logits, xt, t)
        kl = (target * (target.clamp_min(1e-300).log()-prediction.clamp_min(1e-300).log())).sum(-1).mean()
        ce = torch.nn.functional.cross_entropy(logits.float(), x0)
        # t=1 is the terminal reconstruction NLL (equivalent to this KL).
        return (kl + auxiliary*ce).float(), {'posterior_kl': float(kl.detach()), 'x0_ce': float(ce.detach())}
