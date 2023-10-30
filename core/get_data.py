import numpy as np
import torch
from torch_geometric.transforms import Compose, Constant
from core.config import cfg, update_cfg

from torch_geometric.datasets import ZINC, TUDataset
from core.data_utils.exp import PlanarSATPairsDataset
from core.transform import PositionalEncodingTransform, GraphJEPAPartitionTransform


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')


def create_dataset(cfg):
    pre_transform = PositionalEncodingTransform(
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)

    transform_train = transform_eval = None

    if cfg.metis.n_patches > 0:
        _transform_train = GraphJEPAPartitionTransform(n_patches=cfg.metis.n_patches,
                                                metis=cfg.metis.enable,
                                                drop_rate=cfg.metis.drop_rate,
                                                num_hops=cfg.metis.num_hops,
                                                is_directed=cfg.dataset == 'TreeDataset',
                                                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                                patch_num_diff=cfg.pos_enc.patch_num_diff,
                                                num_context=cfg.jepa.num_context, 
                                                num_targets=cfg.jepa.num_targets
                                            )

        _transform_eval = GraphJEPAPartitionTransform(n_patches=cfg.metis.n_patches,
                                            metis=cfg.metis.enable,
                                            drop_rate=0.0,
                                            num_hops=cfg.metis.num_hops,
                                            is_directed=cfg.dataset == 'TreeDataset',
                                            patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                            patch_num_diff=cfg.pos_enc.patch_num_diff,
                                            num_context=cfg.jepa.num_context, 
                                            num_targets=cfg.jepa.num_targets
                                        )
        transform_train = _transform_train
        transform_eval = _transform_eval
    else:
        print('Not supported...')
        exit() 

    if cfg.dataset == 'ZINC':
        root = 'dataset/ZINC'
        train_dataset = ZINC(
            root, subset=True, split='train', pre_transform=pre_transform, transform=transform_train)
        val_dataset = ZINC(root, subset=True, split='val',
                           pre_transform=pre_transform, transform=transform_eval)
        test_dataset = ZINC(root, subset=True, split='test',
                            pre_transform=pre_transform, transform=transform_eval)
    elif cfg.dataset in ['PROTEINS', 'MUTAG', 'DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI']:
        if cfg.dataset not in ['PROTEINS', 'MUTAG', 'DD']:
            pre_transform = Compose([Constant(value=0, cat=False), pre_transform])

        dataset = TUDataset(root='dataset/TUD', name=cfg.dataset, pre_transform=pre_transform)

        return dataset, transform_train, transform_eval

    elif cfg.dataset == 'exp-classify':
        root = "dataset/EXP/"
        dataset = PlanarSATPairsDataset(root, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval

    else:
        print("Dataset not supported.")
        exit(1)

    torch.set_num_threads(cfg.num_workers)
    if not cfg.metis.online:
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("Generating data")

    cfg.merge_from_file('train/configs/zinc.yaml')
    cfg = update_cfg(cfg)
    cfg.metis.n_patches = 0
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    if cfg.dataset == 'exp-classify':
        print('------------Dataset--------------')
        calculate_stats(train_dataset)
        print('------------------------------')
    else:
        print('------------Train--------------')
        calculate_stats(train_dataset)
        print('------------Validation--------------')
        calculate_stats(val_dataset)
        print('------------Test--------------')
        calculate_stats(test_dataset)
        print('------------------------------')
