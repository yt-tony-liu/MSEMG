import os
from argparse import ArgumentParser
import yaml

from dataset import EMGDataset, EMGTestDataset
# from ddpm_1d import GaussianDiffusion1D
from trainer import Trainer1D
# from deep_filter_model import ConditionalModel
from utils import default
from model import EMGMAMBA

# from model_large import EMGMAMBA_large

import wandb
import datetime, pytz

def main(args):
    with open(args.local_cfg, "r") as f:
        file_cfg = yaml.safe_load(f)
    with open(args.experiment_cfg, "r") as f:
        exp_cfg = yaml.safe_load(f)

    # log experiment time
    time = datetime.datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d_%H_%M")
    # if args.train:
    #     exp_cfg['project_name'] = exp_cfg['project_name'] + '_' + time
    exp_cfg['time'] = time

    print(f"Running experiment {exp_cfg['project_name']}")

    dataset_path = file_cfg['sEMG_dataset_dir']
    train_path = os.path.join(dataset_path, 'train')
    validation_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')
    result_path = os.path.join(file_cfg['result_dir'], exp_cfg['project_name'])
    score_path = os.path.join(result_path, f"{exp_cfg['project_name']}.csv")

    
    train_dataset = EMGDataset(train_path)
    validation_dataset = EMGDataset(validation_path)

    model = EMGMAMBA(in_channels = 64, feats=exp_cfg['channels'], n_layer=exp_cfg['mamba_layers'])

    job_type = 'train' if args.train else 'test' if args.test else 'sample'

    if args.train:
        wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="EMGMAMBA",

            # track hyperparameters and run metadata
            config={
            "learning_rate": exp_cfg['lr'],
            "loss_function": exp_cfg['loss_function'],
            "architecture": "Simple",
            "dataset": "NINAPro+MITBIH-NSRD",
            "epochs": exp_cfg['train_epochs'],
            "batch_size": exp_cfg['train_batch_size'],
            "seq_length": exp_cfg['seq_length']
            },
            job_type=job_type
        )


    trainer = Trainer1D(
        model,
        train_dataset = train_dataset,
        validation_dataset = validation_dataset,
        train_epochs = exp_cfg['train_epochs'],
        train_batch_size = exp_cfg['train_batch_size'],
        val_batch_size = exp_cfg['val_batch_size'],
        seq_length = exp_cfg['seq_length'],
        loss_function = exp_cfg['loss_function'],
        train_lr = exp_cfg['lr'],        
        gradient_accumulate_every = exp_cfg['gradient_accumulate_every'],    # gradient accumulation steps
        ema_decay = exp_cfg['ema_decay'],                # exponential moving average decay
        amp = exp_cfg['mix_precision'],
        results_folder = result_path,                      # turn on mixed precision
        num_workers = file_cfg['num_workers'],
        wandb_run = None if not args.train else wandb_run
    )
    inference_milestone = default(exp_cfg['inference_milestone'], None)
    

    

    with open(os.path.join(result_path, 'experiment_cfg.yaml'), 'w') as yaml_file:
        yaml.dump(exp_cfg, yaml_file, default_flow_style=False)

    if args.train:
        print('start training')
        trainer.train()

    if args.test:
        print('testing normal condition')
        test_dataset = EMGTestDataset(test_path)
        trainer.test(test_dataset, score_path, milestone=inference_milestone, save_detail=True)

    if args.sample:  
        print('sampling') 
        file_paths = ['demo file paths'] # path to your inference files
        trainer.denoise_sample(file_paths, milestone=inference_milestone, ddim=exp_cfg['ddim'], denoise_timesteps=exp_cfg['denoise_timesteps'], color='r')




if __name__ == '__main__':
    parser = ArgumentParser(description='train (or resume training) a Diffusion model')
    parser.add_argument('--local_cfg', default='cfg/local_cfg.yaml', help='config for file paths on this machine')
    parser.add_argument('--experiment_cfg', default='cfg/default.yaml', help='experiment setting')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--test_mismatch', action='store_true')
    main(parser.parse_args())