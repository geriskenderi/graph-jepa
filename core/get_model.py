from core.model import GraphJepa
import sys

def create_model(cfg):
    if cfg.dataset == 'ZINC':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 28
        nfeat_edge = 4
        nout = 1  # regression
  
    elif cfg.dataset == 'exp-classify':
        nfeat_node = 2
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'MUTAG':
        nfeat_node = 7
        nfeat_edge = 4
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'PROTEINS':
        nfeat_node = 3
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'DD':
        nfeat_node = 89
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2
    
    elif cfg.dataset == 'REDDIT-BINARY' :
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'REDDIT-MULTI-5K' :
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 5

    elif cfg.dataset == 'IMDB-BINARY' :
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 2

    elif cfg.dataset == 'IMDB-MULTI' :
        nfeat_node = 1
        nfeat_edge = 1
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 3

    if cfg.metis.n_patches > 0:
        if cfg.jepa.enable:
            return GraphJepa(
                nfeat_node=nfeat_node,
                nfeat_edge=nfeat_edge,
                nhid=cfg.model.hidden_size,
                nout=nout,
                nlayer_gnn=cfg.model.nlayer_gnn,
                node_type=node_type,
                edge_type=edge_type,
                nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                gMHA_type=cfg.model.gMHA_type,
                gnn_type=cfg.model.gnn_type,
                rw_dim=cfg.pos_enc.rw_dim,
                lap_dim=cfg.pos_enc.lap_dim,
                pooling=cfg.model.pool,
                dropout=cfg.train.dropout,
                mlpmixer_dropout=cfg.train.mlpmixer_dropout,
                n_patches=cfg.metis.n_patches,
                patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                num_context_patches=cfg.jepa.num_context,
                num_target_patches=cfg.jepa.num_targets
            ) 
        else:
            print('Not supported...')
            sys.exit() 