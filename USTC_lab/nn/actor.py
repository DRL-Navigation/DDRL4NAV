import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from USTC_lab.nn import mlp


class Actor(nn.Module):
    def __init__(self, **kwargs):
        super(Actor, self).__init__()
        pre = kwargs['pre']

        self.pre = pre
        self.device = kwargs['device']

    def _distribution(self, x, play_mode=False):
        raise NotImplemented

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def log_prob_from_distribution(self, pi, act):
        return self._log_prob_from_distribution(pi, act)

    def forward(self, x, act=None, play_mode=False):

        if self.pre:
            x2 = self.pre(x)
        else:
            x2 = x

        pi = self._distribution(x2, play_mode)
        log_p = None
        if act is not None:
            log_p = self._log_prob_from_distribution(pi, act)


        return pi, log_p


class GaussionActor(Actor):
    def __init__(self,
                 action_output_dim=1,
                 device='cpu',
                 soft_max_grid=True,
                 last_input_dim=512,
                 pre=None,
                 nn_dtype=torch.float32):
        super(GaussionActor, self).__init__(
                                            pre=pre,
                                            device=device)
        self.actor_linear = nn.Linear(last_input_dim, action_output_dim)
        log_std = -0.5 * torch.ones(action_output_dim, dtype=nn_dtype)
        self.log_std = torch.nn.Parameter(log_std)

    def _distribution(self, x, play_mode=False):

        mu = self.actor_linear(x)
        if play_mode:
            return mu
        else:
            std = torch.exp(self.log_std)
            pi = Normal(mu, std)

            return pi

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution




class CategoricalActor(Actor):
    def __init__(self, action_output_dim,
                 device='cpu',
                 soft_max_grid=True,
                 last_input_dim=512,
                 pre=None,
                 nn_dtype=torch.float32):
        super(CategoricalActor, self).__init__(
                                               pre=pre,
                                               device=device,)
        self.logits_net = None
        self.action_output_dim = action_output_dim
        self.actor_linear = nn.Linear(last_input_dim, action_output_dim)
        self.soft_max_grid = soft_max_grid

    def _distribution(self, x, play_mode=False):
        x = self.actor_linear(x)
        # TODO soft max grid
        #if self.soft_max_grid:
        x = F.softmax(x, dim=-1)
        if play_mode:
            return x
        else:
            return Categorical(x)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
