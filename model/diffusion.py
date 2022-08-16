import math
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion class
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        schedule='linear',
        n_timestep=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        device='cuda'
    ):
        super().__init__()
        self.num_timesteps = n_timestep
        self.device=device


        # set noise schedule
        # all variables have length of num_timesteps + 1
        betas = torch.zeros(n_timestep + 1, dtype=torch.float32, device=device)
        if schedule == 'linear':
            betas[1:] = torch.linspace(linear_start, linear_end, n_timestep, device=device, dtype=torch.float32)
        elif schedule == 'quad':
            betas[1:] = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, device=device, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError

        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        # standard deviation

        self.calculate_p_coeffs()
        self.calculate_coeffs_conditional()

        # plt.plot(self.betas)
        # plt.plot(self.alpha_bar)
        # plt.plot(self.sqrt_delta**2)
        # plt.plot(self.sqrt_delta_estimated**2)
        # plt.legend(['beta', 'alpha_bar', 'delta', 'delta_estimated'])
        # plt.show(block=True)

        # plt.plot(self.c_xt[1:])
        # plt.plot(self.c_yt[1:])
        # plt.plot(self.c_epst[1:])
        # plt.legend(['c_xt', 'c_yt', 'c_epst'])
        # plt.show(block=True)

    def calculate_p_coeffs(self):

        # for infer
        sigma = torch.zeros_like(self.betas)
        sigma[1:] = ((1.0 - self.alpha_bar[:-1]) / (1.0 - self.alpha_bar[1:]) * self.betas[1:]) ** 0.5
        predicted_noise_coeff = torch.zeros_like(self.betas)
        predicted_noise_coeff[1:] = self.betas[1:]/ torch.sqrt(1-self.alpha_bar[1:])

        self.register_buffer('predicted_noise_coeff', predicted_noise_coeff)
        self.register_buffer('sigma', sigma)
        # Supportive Parameters
        supportive_gamma = torch.zeros_like(self.betas)
        supportive_gamma[1] = 0.2
        supportive_gamma[2:] = sigma[2:]

        supportive_sigma_hat = torch.zeros_like(self.betas)
        supportive_sigma_hat[1:] = sigma[1:] - supportive_gamma[1:]/torch.sqrt(self.alphas[1:])

        self.register_buffer('supportive_gamma', supportive_gamma)
        self.register_buffer('supportive_sigma_hat', supportive_sigma_hat)

    def calculate_coeffs_conditional(self):
        # q process
        m = torch.sqrt((1 - self.alpha_bar) / self.sqrt_alpha_bar)
        m[-1] = 1
        self.register_buffer('m', m)
        # variance
        # 1 - alhpa_bar * ( 1+m^2)
        delta = (1 - self.alpha_bar) - m ** 2 * self.alpha_bar
        # standard deviation

        time_steps = torch.arange(0, len(self.betas))
        self.register_buffer('sqrt_delta', torch.sqrt(delta))


        # p process
        # (1 - m_t )/ (1 - m_{t-1})
        one_minus_m_ratio = (1 - m[1:]) / (1 - m[:-1])
        # alpha_t * delta_{t-1}
        alpha_t_delta_t_1 = self.alphas[1:] * delta[:-1]
        # delta_{t|t-1}
        delta_t_given_t_1 = delta[1:] - one_minus_m_ratio ** 2 * alpha_t_delta_t_1
        sqrt_alphas = torch.sqrt(self.alphas[1:])

        c_xt = torch.zeros_like(self.betas)
        c_xt[1:] = one_minus_m_ratio * delta[:-1] / delta[1:] * sqrt_alphas \
                   + (1-self.m[:-1]) * (delta_t_given_t_1 / delta[1:]) * (1 / sqrt_alphas)

        c_yt = torch.zeros_like(self.betas)
        c_yt[1:] = (self.m[:-1] * delta[1:] - self.m[1:] * one_minus_m_ratio * alpha_t_delta_t_1) \
                   * self.sqrt_alpha_bar[:-1] / delta[1:]

        c_epst = torch.zeros_like(self.betas)
        c_epst[1:] = (1 - self.m[:-1]) * delta_t_given_t_1 / delta[1:] \
                     * torch.sqrt(1-self.alpha_bar[1:]) / sqrt_alphas

        # estimated variance
        delta_estimated = torch.zeros_like(self.betas)
        # delta_estimated[1:] = delta_t_given_t_1 * delta[1:] / delta[:-1]
        delta_estimated[1:] = delta_t_given_t_1 * delta[:-1] / delta[1:]

        self.register_buffer('c_xt', c_xt)
        self.register_buffer('c_yt', c_yt)
        self.register_buffer('c_epst', c_epst)
        self.register_buffer('sqrt_delta_estimated', torch.sqrt(delta_estimated))


    @torch.no_grad()
    def p_transition_sr3(self, x_t, t, predicted):
        """
        sr3 p_transition
        noise variance is different from Ho et al 2020
        """
        x_t_1 = (x_t - self.predicted_noise_coeff[t] * predicted)/(self.alphas[t])**0.5
        if t > 1:
            noise = torch.randn_like(x_t)
            x_t_1 += torch.sqrt(self.betas[t]) * noise

        return x_t_1.clamp_(-1., 1.)

    @torch.no_grad()
    def p_transition(self, x_t, t, predicted):
        """
        p_transition from Ho et al 2020, and wavegrad, conditioned on t, t is scalar
        """

        # mean
        x_t_1 = (x_t - self.predicted_noise_coeff[t] * predicted)/(self.alphas[t])**0.5
        # add gaussian noise with std of sigma
        if t > 1:
            noise = torch.randn_like(x_t)
            x_t_1 += self.sigma[t] * noise

        return x_t_1.clamp_(-1., 1.)

    @torch.no_grad()
    def p_transition_supportive(self, x_t, t, predicted_noise, condition):
        """
        supportive transition from Lu et al 2021
        x_t is the sample at t
        t is the time step
        predicted_noise is the noise predicted by the denoising model
        condition is the conditional input
        """

        # mean
        mu_t = (x_t - self.predicted_noise_coeff[t] * predicted_noise)
        # add gaussian noise with std of sigma
        x_t_1 = ((1-self.supportive_gamma[t]) * mu_t + self.supportive_gamma[t] * condition)/(self.alphas[t])**0.5
        if t > 1:
            noise = torch.randn_like(x_t)
            x_t_1 += max(0, self.supportive_sigma_hat[t]) * noise
        return x_t_1.clamp_(-1., 1.)

    @torch.no_grad()
    def p_transition_conditional(self, x_t, t, predicted_noise, condition):
        """
        conditional p transition
        """
        mean = self.c_xt[t] * x_t + self.c_yt[t] * condition - self.c_epst[t] * predicted_noise
        x_t_1 = mean
        if t > 1:
            # add gaussian noise portion
            noise = torch.randn_like(x_t)
            x_t_1 = x_t_1 + self.sqrt_delta_estimated[t] * noise

        return x_t_1.clamp_(-1., 1.)

    def q_stochastic(self, x_0, noise, t_is_integer=False):
        """
        x_0 has shape of [B, 1, T]
        choose a random diffusion step to calculate loss
        """
        # 0 dim is the batch size
        b = x_0.shape[0]
        alpha_bar_sample_shape = torch.ones(x_0.ndim, dtype=torch.int)
        alpha_bar_sample_shape[0] = b

        # choose random step for each one in this batch
        # change to torch
        t = torch.randint(1, self.num_timesteps + 1, [b], device=x_0.device)
        if t_is_integer:
            sqrt_alpha_bar_sample = self.sqrt_alpha_bar[t]
            random_step = 0
        else:
            # sample noise level using uniform distribution
            l_a, l_b = self.sqrt_alpha_bar[t - 1], self.sqrt_alpha_bar[t]
            random_step =  torch.rand(b, device=x_0.device)
            sqrt_alpha_bar_sample = l_a + random_step * (l_b - l_a)

        sqrt_alpha_bar_sample = sqrt_alpha_bar_sample.view(tuple(alpha_bar_sample_shape))

        x_t = sqrt_alpha_bar_sample * x_0 + torch.sqrt((1. - torch.square(sqrt_alpha_bar_sample))) * noise

        return x_t, sqrt_alpha_bar_sample, (t+random_step).view(tuple(alpha_bar_sample_shape))

    def q_stochastic_conditional(self, x_0, y, noise):
        """
            x_0 is the training target
            y is the conditional input
        """

        # 0 dim is the batch size
        b = x_0.shape[0]
        noise_level_sample_shape = torch.ones(x_0.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b

        # choose random step [1, num_timesteps] for each one in this batch
        t = torch.randint(1, self.num_timesteps + 1, tuple(noise_level_sample_shape), device=x_0.device)

        sqrt_alpha_bar_sample = self.sqrt_alpha_bar[t]

        # sqrt(delta_t) * eps
        gaussian_noise = self.sqrt_delta[t] * noise
        # noise from sample
        noise_from_condition = self.m[t] * self.sqrt_alpha_bar[t] * (y - x_0)

        x_t = self.sqrt_alpha_bar[t] * x_0 + noise_from_condition + gaussian_noise

        combined_noise = 1. / (torch.sqrt(1. - self.alpha_bar[t])) * (noise_from_condition + gaussian_noise)

        # use sqrt_alpha_bar as condition
        return x_t, combined_noise, sqrt_alpha_bar_sample

    def get_x_T(self, condition):

        # 0 dim is the batch size

        noise = torch.randn_like(condition, device=condition.device)
        b = condition.shape[0]
        noise_level_sample_shape = torch.ones(condition.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b

        t = self.num_timesteps * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
        t = t.type(torch.LongTensor)

        sqrt_alpha_bar_sample = self.sqrt_alpha_bar[t]

        sqrt_alpha_bar_sample = sqrt_alpha_bar_sample.view(tuple(noise_level_sample_shape))

        x_T = sqrt_alpha_bar_sample * condition + torch.sqrt((1. - torch.square(sqrt_alpha_bar_sample))) * noise

        # use sqrt_alpha_bar as condition
        return x_T

    def get_x_T_conditional(self, condition):

        # 0 dim is the batch size

        noise = torch.randn_like(condition, device=condition.device)
        b = condition.shape[0]
        noise_level_sample_shape = torch.ones(condition.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b

        t = self.num_timesteps* torch.ones(tuple(noise_level_sample_shape), device=condition.device)
        t = t.type(torch.LongTensor)

        sqrt_alpha_bar_sample = self.sqrt_alpha_bar[t]
        sqrt_alpha_bar_sample = sqrt_alpha_bar_sample.view(tuple(noise_level_sample_shape))

        x_T = sqrt_alpha_bar_sample * condition + self.sqrt_delta[t] * noise

        # use sqrt_alpha_bar as condition
        return x_T

    def get_noise_level(self, t):
        """
        noise level is sqrt alpha bar
        """
        return self.sqrt_alpha_bar[t]


if __name__ == '__main__':
    diffussion = GaussianDiffusion(device='cpu')
    y_0 = torch.ones([2,1, 3])
    noise = torch.randn_like(y_0)
    diffussion.q_stochastic(y_0, noise)

    predicted = noise
    y_t = y_0
    diffussion.p_transition(y_t, 0, predicted)
    diffussion.p_transition(y_t, 1, predicted)
