import wandb


def log_training_info(info, use_wandb=False):
    print("")
    for k, v in info.items():
        s = "{k: <30}{v:0.5f}".format(k=k, v=v)
        print(s)

    if use_wandb:
        wandb.log(info)
