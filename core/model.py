import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter
from einops.layers.torch import Rearrange
import core.model_utils.gMHA_wrapper as gMHA_wrapper

from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN

class GraphJepa(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 num_context_patches=1,
                 num_target_patches=4):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.n_patches = n_patches
        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim
        self.nhid = nhid
        self.nfeat_edge = nfeat_edge
        self.num_context_patches=num_context_patches
        self.num_target_patches=num_target_patches

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.context_encoder = getattr(gMHA_wrapper, 'Standard')(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)
        self.target_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        # Predictor
        self.target_predictor = MLP(
            nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

        # Use this predictor if you wish to do euclidean or poincaré embeddings in the latent space
        # self.target_predictor = MLP(
        #     nhid, 2, nlayer=3, with_final_activation=False, with_norm=False)

    def forward(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='max')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)


        ######################## Graph-JEPA ########################
        # Create the correct indexer for each subgraph given the batching procedure
        batch_indexer = torch.tensor(np.cumsum(data.call_n_patches))
        batch_indexer = torch.hstack((torch.tensor(0), batch_indexer[:-1])).to(data.y.device)

        # Get idx of context and target subgraphs according to masks
        context_subgraph_idx = data.context_subgraph_idx + batch_indexer
        target_subgraphs_idx = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs]).to(data.y.device)
        target_subgraphs_idx += batch_indexer.unsqueeze(1)

        # Get context and target subgraph (mpnn) embeddings
        context_subgraphs = subgraph_x[context_subgraph_idx]
        target_subgraphs = subgraph_x[target_subgraphs_idx.flatten()]

        # Construct context and target PEs frome the node pes of each subgraph
        target_pes = patch_pes[target_subgraphs_idx.flatten()]
        context_pe = patch_pes[context_subgraph_idx] 
        context_subgraphs += self.patch_rw_encoder(context_pe)
        encoded_tpatch_pes = self.patch_rw_encoder(target_pes)
        
        # Prepare inputs for MHA
        target_x = target_subgraphs.reshape(-1, self.num_target_patches, self.nhid)
        context_x = context_subgraphs.unsqueeze(1)

        # Given that there's only one element the attention operation "won't do anything"
        # This is simply for commodity of the EMA between context and target encoders
        context_mask = data.mask.flatten()[context_subgraph_idx].reshape(-1, self.num_context_patches) # this should be -1 x num context
        context_x = self.context_encoder(context_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~context_mask)

        # The target forward step musn't store gradients, since the target encoder is optimized via EMA
        with torch.no_grad():
            if hasattr(data, 'coarsen_adj'):
                subgraph_incides = torch.vstack([torch.tensor(dt) for dt in data.target_subgraph_idxs])
                patch_adj = data.coarsen_adj[
                    torch.arange(target_x.shape[0]).unsqueeze(1).unsqueeze(2),  # Batch dimension
                    subgraph_incides.unsqueeze(1),  # Row dimension
                    subgraph_incides.unsqueeze(2)   # Column dimension
                ]
                target_x = self.target_encoder(target_x, patch_adj, None)
            else:
                target_x = self.target_encoder(target_x, None, None)

            # Predict the coordinates of the patches in the Q1 hyperbola
            # Remove this part if you wish to do euclidean or poincaré embeddings in the latent space
            x_coord = torch.cosh(target_x.mean(-1).unsqueeze(-1))
            y_coord = torch.sinh(target_x.mean(-1).unsqueeze(-1))
            target_x = torch.cat([x_coord, y_coord], dim=-1)


        # Make predictions using the target predictor: for each target subgraph, we use the context + the target PE
        target_prediction_embeddings = context_x + encoded_tpatch_pes.reshape(-1, self.num_target_patches, self.nhid)
        target_y = self.target_predictor(target_prediction_embeddings)
       
        return target_x, target_y

    def encode(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)
        
        # Handles different patch sizes based on the data object for multiscale training
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0][0], -1)

        # Eval via target encoder
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj if hasattr(
                                        data, 'coarsen_adj') else None, ~data.mask) # Don't attend to empty patches when doing the final encoding
        
        # Global Average Pooling
        out = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / data.mask.sum(1, keepdim=True)
        return out

    def encode_nopool(self, data):
        x = self.input_encoder(data.x).squeeze()

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1)).float().unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        edge_index = data.combined_subgraphs
        e = edge_attr[data.subgraphs_edges_mapper]
        batch_x = data.subgraphs_batch
        pes = data.rw_pos_enc[data.subgraphs_nodes_mapper]
        patch_pes = scatter(pes, batch_x, dim=0, reduce='mean')
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x += self.patch_rw_encoder(patch_pes)
        

        # Eval via target encoder
        mixer_x = subgraph_x.reshape(len(data.call_n_patches), data.call_n_patches[0], -1)
        mixer_x = self.target_encoder(mixer_x, data.coarsen_adj if hasattr(
                                        data, 'coarsen_adj') else None, ~data.mask) # Don't attend to empty patches when doing the final encoding
        
        return mixer_x