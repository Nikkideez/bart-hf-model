
search_method = "bayes" 

sweep_config = {
    'method': search_method,
    'metric': {
       'name': 'eval/spot_acc',
       'goal': 'maximize'
    }
}


# hyperparameter range
parameters_dict = {
    'epochs': {
        'value': 50
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 2e-5,
        'max': 6e-4
    },
    'weight_decay': {
        'values': [0.3, 0.4, 0.5, 0.6]
    },
    'gradient_accumulation_steps': {
        'values': [1, 2, 4, 6]
    },
}

