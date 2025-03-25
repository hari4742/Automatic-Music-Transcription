from omegaconf import OmegaConf


def get_sweep_config(wandb):
    # Convert sweep parameters to nested structure matching Hydra config
    sweep_config = OmegaConf.create({
        'training': {
            'lr': wandb.config.lr,
            'batch_size': wandb.config.batch_size,
            'weight_decay': wandb.config.weight_decay
        },
        'model': {
            'kernel1_size_x': wandb.config.kernel1_size_x,
            'kernel1_size_y': wandb.config.kernel1_size_y,
            'out_channels1': wandb.config.out_channels1,
            'max_pool_kernel1_x': wandb.config.max_pool_kernel1_x,
            'kernel2_size_x': wandb.config.kernel2_size_x,
            'out_channels2': wandb.config.out_channels2,
            'lstm1_hidden_state': wandb.config.lstm1_hidden_state,
            'dropout_size': wandb.config.dropout_size,
            'lstm2_hidden_state': wandb.config.lstm2_hidden_state
        }
    })

    return sweep_config
