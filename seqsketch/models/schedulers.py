from diffusers.schedulers import DDIMScheduler


class myDDIMScheduler(DDIMScheduler):
    def __init__(self, params):
        super().__init__(
            num_train_timesteps=params.num_train_timesteps,
            beta_start=params.beta_start,
            beta_end=params.beta_end,
            beta_schedule=params.beta_schedule,
            clip_sample=params.clip_sample,
            prediction_type = params.prediction_type
        )
