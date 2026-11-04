import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()

# -------------------------
# Basic settings
# -------------------------
parser.add_argument('--root_path', type=str,
                    default='data/TEP', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='TEP', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='lists/lists_TEP', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

# -------------------------
# Loss settings
# -------------------------
parser.add_argument(
    '--loss_mode',
    type=str,
    default='ce_dice',
    choices=[
        'ce_dice',
        'weighted_ce_dice',
        'ce_dice_boundary',
        'ce_dice_boundary_cldice'
    ],
    help='Which loss combination to use'
)

parser.add_argument(
    '--class_weights',
    type=float,
    nargs='+',
    default=None,
    help='Class weights for weighted CE. Example: --class_weights 0.1 1.0 1.0 1.5 1.5'
)

parser.add_argument(
    '--ce_weight',
    type=float,
    default=0.5,
    help='Weight for CE loss'
)

parser.add_argument(
    '--dice_weight',
    type=float,
    default=0.5,
    help='Weight for Dice loss'
)

parser.add_argument(
    '--boundary_weight',
    type=float,
    default=0.2,
    help='Weight for Boundary loss'
)

parser.add_argument(
    '--cldice_weight',
    type=float,
    default=0.1,
    help='Weight for clDice auxiliary loss'
)

parser.add_argument(
    '--cldice_classes',
    type=int,
    nargs='+',
    default=[1, 4],
    help='Class ids for clDice auxiliary loss. Example: --cldice_classes 1 4'
)

parser.add_argument(
    '--cldice_iter',
    type=int,
    default=10,
    help='Soft skeletonization iterations for clDice'
)

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'TEP': {
            'root_path': 'TEP_dataset/ro',
            'list_dir': 'lists/lists_TEP/ro',
            'num_classes': 5
        },
    }

    if dataset_name not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # weighted_ce_dice 모드일 때 class_weights 길이 체크
    if args.loss_mode == 'weighted_ce_dice':
        if args.class_weights is None:
            raise ValueError(
                "loss_mode='weighted_ce_dice' requires --class_weights. "
                "Example: --class_weights 0.1 1.0 1.0 1.5 1.5"
            )
        if len(args.class_weights) != args.num_classes:
            raise ValueError(
                f"--class_weights length must be {args.num_classes}, "
                f"but got {len(args.class_weights)}"
            )

    # clDice 모드일 때 class id 체크
    if args.loss_mode == 'ce_dice_boundary_cldice':
        if args.cldice_classes is None or len(args.cldice_classes) == 0:
            raise ValueError(
                "loss_mode='ce_dice_boundary_cldice' requires --cldice_classes. "
                "Example: --cldice_classes 1 4"
            )
        for c in args.cldice_classes:
            if c < 0 or c >= args.num_classes:
                raise ValueError(f"Invalid clDice class id: {c}")

    args.exp = 'TU_' + dataset_name + str(args.img_size)

    snapshot_path = "260406/cw_ce_d/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # loss mode 정보를 실험명에 추가
    snapshot_path = snapshot_path + '_' + args.loss_mode

    # weighted CE면 weight 정보도 경로에 추가
    if args.loss_mode == 'weighted_ce_dice':
        weight_str = '-'.join([str(w) for w in args.class_weights])
        snapshot_path = snapshot_path + '_cw_' + weight_str

    # boundary, clDice 가중치도 구분 가능하게 추가
    if args.loss_mode in ['ce_dice_boundary', 'ce_dice_boundary_cldice']:
        snapshot_path = snapshot_path + f'_bw{args.boundary_weight}'

    if args.loss_mode == 'ce_dice_boundary_cldice':
        class_str = '-'.join([str(c) for c in args.cldice_classes])
        snapshot_path = snapshot_path + f'_clw{args.cldice_weight}_clc{class_str}'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    net = ViT_seg(
        config_vit,
        img_size=args.img_size,
        num_classes=config_vit.n_classes
    ).cuda()

    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'TEP': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)
