
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torchvision.utils import save_image, make_grid
from domainbed.networks import load_munit_model


from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, MetaDataLoader, MetaFastDataLoader

import tensorboardX
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data/")
    parser.add_argument('--dataset', type=str, default="NYPD")
    parser.add_argument('--algorithm', type=str, default="MBDG_Meta")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="/home/kxj200023/data/ieee2024/test")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--holdout_fraction_finetune', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--step', type=int, default=2,
                        help='Cotrain:2, Pretrain:3')
    args = parser.parse_args()


    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args.test_envs, args.step)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed), args.test_envs)
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError


    in_splits = []
    out_splits = []
    meta_train_splits = []
    finetune_train_splits = []
    finetune_test_splits = []
    for env_i, env in enumerate(dataset):

        if env_i in args.test_envs:
            in_, out = misc.split_dataset(env,
              int(len(env) * args.holdout_fraction_finetune),
              misc.seed_hash(args.trial_seed, env_i))
            finetune_train_splits.append(in_)
            finetune_test_splits.append(out)
        else:
            out, in_ = misc.split_dataset(env,
              int(len(env) * args.holdout_fraction),
              misc.seed_hash(args.trial_seed, env_i))
            meta_train_splits.append(in_)

        in_splits.append(in_)
        out_splits.append(out)
    merged_meta_train_splits = misc._MergeDataset(meta_train_splits)

    if args.algorithm == 'MBDG_Meta' or args.algorithm == 'MBDG_Meta_Ab1':
        meta_support_loader = MetaDataLoader(
            dataset=merged_meta_train_splits,
            num_shots=hparams['k_shot'],
            num_workers=dataset.N_WORKERS)
        meta_query_loader = MetaDataLoader(
            dataset=merged_meta_train_splits,
            num_shots=hparams['k_query'],
            num_workers=dataset.N_WORKERS)
        train_loaders = [meta_support_loader, meta_query_loader]
    else:
        train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for i, env in enumerate(in_splits)
            if i not in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        # num_shots=64,
        batch_size=1024,
        num_workers=dataset.N_WORKERS)
        for env in (in_splits + out_splits)]

    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)


    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['k_shot'] for env in meta_train_splits])

    n_meta_steps = args.steps or dataset.N_STEPS
    meta_checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    meta_train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_dir + "/logs/meta", "tensorboard"))


    def write_loss(iterations, trainer, train_writer):
        train_writer.add_scalar("loss", trainer.loss, iterations + 1)
        train_writer.add_scalar("l_cls", trainer.l_cls, iterations + 1)
        if hasattr(trainer,'l_inv'):
            train_writer.add_scalar("l_inv", trainer.l_inv, iterations + 1)
        if hasattr(trainer,'l_fair'):
            train_writer.add_scalar("l_fair", trainer.l_fair, iterations + 1)
        train_writer.add_scalar("dual_var1", trainer.dual_var1, iterations + 1)
        train_writer.add_scalar("dual_var2", trainer.dual_var2, iterations + 1)


    def draw_loss_curve(tensorboard_dir, save_path):
        log_dir = tensorboard_dir
        event_file = [f for f in os.listdir(log_dir) if f.startswith('events')][-1]
        event_path = os.path.join(log_dir, event_file)

        event_acc = EventAccumulator(event_path)
        event_acc.Reload()

        losses = event_acc.Scalars('loss')
        steps = [e.step for e in losses]
        loss_values = [e.value for e in losses]
        l_cls_values = [e.value for e in event_acc.Scalars('l_cls')]
        l_inv_values = [e.value for e in event_acc.Scalars('l_inv')]
        l_fair_values = [e.value for e in event_acc.Scalars('l_fair')]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, loss_values, label='Loss')
        plt.plot(steps, l_cls_values, label='Loss_cls')
        plt.plot(steps, l_inv_values, label='Loss_inv')
        plt.plot(steps, l_fair_values, label='Loss_fair')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(save_path)

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
        algorithm.to(device)


    def check_for_nan(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in parameter: {name}")
                return True
        return False

    print('Start Meta Training')
    last_results_keys = None
    for step in range(start_step, n_meta_steps):
        step_start_time = time.time()
        minibatches_device = [[(x.to(device), y.to(device), z.to(device))
        for x, y, z in next(train_minibatches_iterator)] for _ in range(hparams['num_task'])]
        step_vals = algorithm.meta_train(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        write_loss(step, algorithm, meta_train_writer)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % meta_checkpoint_freq == 0) or (step == n_meta_steps - 1):
        # if False:
        #     check_for_nan(algorithm)
            value_step_start_time = time.time()
            metric_names = ['acc', 'dp', 'Δdp', 'eopp', 'Δeopp', 'Δeo', 'auc']
            results_dict = {}
            for metric in metric_names:
                results_dict[metric] = {'step': step,
                'epoch': step / steps_per_epoch,}
                for key, val in checkpoint_vals.items():
                    results_dict[metric][key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders)
            for name, loader in evals:
                temp_result = misc.metrics(algorithm, loader, device)
                for metric in metric_names:
                    results_dict[metric][name+'_'+metric] = temp_result[metric]
            for metric in metric_names:
                temp_keys = sorted(results_dict[metric].keys())
                misc.print_row(temp_keys, colwidth=14) # print name of each column
                last_results_keys = temp_keys
                misc.print_row([results_dict[metric][key] for key in temp_keys],
                    colwidth=14)

            print('valuate time:', time.time() - value_step_start_time)
            print()
            for metric in metric_names:
                results_dict[metric].update({
                    'hparams': hparams,
                    'args': vars(args)
                })

            epochs_path = os.path.join(args.output_dir, 'meta_results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results_dict, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_meta_step{step}.pkl')
    save_checkpoint('model_meta.pkl')

    with open(os.path.join(args.output_dir, 'meta_done'), 'w') as f:
        f.write('meta_done')

    meta_train_writer.close()

    print('Start Fine Tune')
    # algorithm_dict = torch.load(os.path.join(args.output_dir, 'model_meta.pkl'))
    # algorithm.load_state_dict(algorithm_dict['model_dict'])
    # Finetune
    finetune_train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env in finetune_train_splits]

    finetune_minibatches_iterator = zip(*finetune_train_loaders)

    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = min([len(env) / hparams['batch_size'] for env in finetune_train_splits])
    n_steps = args.steps or dataset.N_STEPS
    finetune_checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    finetune_writer = tensorboardX.SummaryWriter(os.path.join(args.output_dir + "/logs/finetune", "tensorboard"))
    start_step = 0
    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device), z.to(device))
        for x, y, z in next(finetune_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        write_loss(step, algorithm, finetune_writer)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        check_for_nan(algorithm)

        if (step % finetune_checkpoint_freq == 0) or (step == n_steps - 1):
        # if False:
            value_step_start_time = time.time()
            metric_names = ['acc', 'dp', 'Δdp', 'eopp', 'Δeopp', 'Δeo', 'auc']
            results_dict = {}
            for metric in metric_names:
                results_dict[metric] = {'step': step,
                                        'epoch': step / steps_per_epoch, }
                for key, val in checkpoint_vals.items():
                    results_dict[metric][key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders)
            for name, loader in evals:
                temp_result = misc.metrics(algorithm, loader, device)
                for metric in metric_names:
                    results_dict[metric][name + '_' + metric] = temp_result[metric]
            for metric in metric_names:
                temp_keys = sorted(results_dict[metric].keys())
                misc.print_row(temp_keys, colwidth=14)  # print name of each column
                last_results_keys = temp_keys
                misc.print_row([results_dict[metric][key] for key in temp_keys],
                               colwidth=14)
            print('valuate time:', time.time() - value_step_start_time)
            print()

            for metric in metric_names:
                results_dict[metric].update({
                    'hparams': hparams,
                    'args': vars(args)
                })

            epochs_path = os.path.join(args.output_dir, 'finetune_results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results_dict, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_finetune_step{step}.pkl')

    save_checkpoint('model_finetune.pkl')

    with open(os.path.join(args.output_dir, 'finetune_done'), 'w') as f:
        f.write('finetune_done')

    finetune_writer.close()

    draw_loss_curve(os.path.join(args.output_dir, "logs/meta/tensorboard"),
                    os.path.join(args.output_dir, "loss_curve_meta.png"))
    draw_loss_curve(os.path.join(args.output_dir, "logs/finetune/tensorboard"),
                    os.path.join(args.output_dir, "loss_curve_finetune.png"))