import torch
from torch import nn
from base import BaseModel
from .diffusion import GaussianDiffusion
from tqdm import tqdm

class SDDM(BaseModel):
    def __init__(self, diffusion:GaussianDiffusion, noise_estimate_model:nn.Module,
                 noise_condition='sqrt_alpha_bar', p_transition='original', q_transition='original'):
        super().__init__()
        self.diffusion = diffusion
        self.noise_estimate_model = noise_estimate_model
        self.num_timesteps = self.diffusion.num_timesteps
        self.noise_condition = noise_condition
        self.p_transition = p_transition
        self.q_transition = q_transition
        if noise_condition != 'sqrt_alpha_bar' and noise_condition != 'time_step':
            raise NotImplementedError

        if p_transition != 'original' and p_transition != 'supportive' \
                and p_transition != 'sr3' and p_transition != 'conditional'\
                and p_transition != 'condition_in':
            raise NotImplementedError

        if q_transition != 'original' and q_transition != 'conditional':
            raise NotImplementedError

    # train step
    def forward(self, target, condition):
        """
        target is the target source
        condition is the noisy conditional input
        """

        # generate noise
        if self.q_transition == 'original':
            noise = torch.randn_like(target, device=target.device)
            x_t, noise_level, t = self.diffusion.q_stochastic(target, noise)
            if self.noise_condition == 'sqrt_alpha_bar':
                predicted = self.noise_estimate_model(condition, x_t, noise_level)
            elif self.noise_condition == 'time_step':
                predicted = self.noise_estimate_model(condition, x_t, t)
        elif self.q_transition == 'conditional':
            noise = torch.randn_like(target, device=target.device)
            x_t, noise, noise_level = self.diffusion.q_stochastic_conditional(target, condition, noise)
            predicted = self.noise_estimate_model(condition, x_t, noise_level)

        return predicted, noise

    @torch.no_grad()
    def infer(self, condition, continuous=False):
        # condition is audio
        # initial input

        # TODO: predict noise level to reduce computation cost

        if self.p_transition == 'conditional':
            # start from conditional input + gaussian noise, conditional diffusion process
            x_t = self.diffusion.get_x_T_conditional(condition)
        elif self.p_transition == 'condition_in':
            # start from conditional input + gaussian noise, original diffusion process
            x_t = self.diffusion.get_x_T(condition)
        elif self.p_transition == 'supportive':
            # start from conditional input + gaussian noise, original diffusion process
            x_t = condition
        else:
            # start from total noise
            x_t = torch.randn_like(condition, device=condition.device)


        num_timesteps = self.diffusion.num_timesteps
        sample_inter = (1 | (num_timesteps // 100))

        batch_size = condition.shape[0]
        b = condition.shape[0]
        noise_level_sample_shape = torch.ones(condition.ndim, dtype=torch.int)
        noise_level_sample_shape[0] = b
        # iterative refinement
        if continuous:
            assert batch_size==1, 'Batch size must be 1 to do continuous sampling'
            samples = [condition]
            for t in tqdm(reversed(range(1, self.num_timesteps+1)), desc='sampling loop time step', total=self.num_timesteps):
                if self.noise_condition == 'sqrt_alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=condition.device)
                    predicted = self.noise_estimate_model(condition, x_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
                    predicted = self.noise_estimate_model(condition, x_t, time_steps)

                if self.p_transition == 'original' or self.p_transition == 'condition_in':
                    x_t = self.diffusion.p_transition(x_t, t, predicted)
                elif self.p_transition == 'sr3':
                    x_t = self.diffusion.p_transition_sr3(x_t, t, predicted)
                elif self.p_transition == 'supportive':
                    x_t = self.diffusion.p_transition_supportive(x_t, t, predicted, condition)
                elif self.p_transition == 'conditional':
                    x_t = self.diffusion.p_transition_conditional(x_t, t, predicted, condition)

                if t % sample_inter == 0:
                    samples.append(x_t)

            return samples

        else:
            for t in reversed(range(1, self.num_timesteps+1)):
                if self.noise_condition == 'sqrt_alpha_bar':
                    noise_level = self.diffusion.get_noise_level(t) * torch.ones(tuple(noise_level_sample_shape),
                                                                                 device=condition.device)
                    predicted = self.noise_estimate_model(condition, x_t, noise_level)
                elif self.noise_condition == 'time_step':
                    time_steps = t * torch.ones(tuple(noise_level_sample_shape), device=condition.device)
                    predicted = self.noise_estimate_model(condition, x_t, time_steps)

                if self.p_transition == 'original' or self.p_transition == 'condition_in':
                    x_t = self.diffusion.p_transition(x_t, t, predicted)
                elif self.p_transition == 'sr3':
                    x_t = self.diffusion.p_transition_sr3(x_t, t, predicted)
                elif self.p_transition == 'supportive':
                    x_t = self.diffusion.p_transition_supportive(x_t, t, predicted, condition)
                elif self.p_transition == 'conditional':
                    x_t = self.diffusion.p_transition_conditional(x_t, t, predicted, condition)

            return x_t

