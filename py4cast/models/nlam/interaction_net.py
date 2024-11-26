import torch
import torch_geometric as pyg
from torch import nn

from py4cast.models.base import CheckpointWrapper


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of a generic Interaction Network, from Battaglia et al. (2016)
    """

    def __init__(
        self,
        edge_index,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
        checkpoint=False,
    ):
        """
        Create a new InteractionNet

        edge_index: (2,M), Edges in pyg format
        input_dim: Dimensionality of input representations, for both nodes and edges
        update_edges: If new edge representations should be computed and returned
        hidden_layers: Number of hidden layers in MLPs
        hidden_dim: Dimensionality of hidden layers, if None then same as input_dim
        edge_chunk_sizes: List of chunks sizes to split edge representation into and
            use separate MLPs for (None = no chunking, same MLP)
        aggr_chunk_sizes: List of chunks sizes to split aggregated node representation
            into and use separate MLPs for (None = no chunking, same MLP)
        aggr: Message aggregation method (sum/mean)
        """
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr)

        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim

        # Make both sender and receiver indices of edge_index start at 0
        edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]
        # Store number of receiver nodes according to edge_index
        self.num_rec = edge_index[1].max() + 1
        edge_index[0] = edge_index[0] + self.num_rec  # Make sender indices after rec
        self.register_buffer("edge_index", edge_index, persistent=False)

        # Create MLPs
        edge_mlp_recipe = [3 * input_dim] + [hidden_dim] * (hidden_layers + 1)
        aggr_mlp_recipe = [2 * input_dim] + [hidden_dim] * (hidden_layers + 1)

        if edge_chunk_sizes is None:
            self.edge_mlp = make_mlp(edge_mlp_recipe, checkpoint=checkpoint)
        else:
            # Not test : split was never used during dev phase
            self.edge_mlp = SplitMLPs(
                [
                    make_mlp(edge_mlp_recipe, checkpoint=checkpoint)
                    for _ in edge_chunk_sizes
                ],
                edge_chunk_sizes,
            )
            if checkpoint:
                self.edge_mlp = CheckpointWrapper(self.edge_mlp)

        if aggr_chunk_sizes is None:
            self.aggr_mlp = make_mlp(aggr_mlp_recipe, checkpoint=checkpoint)
        else:
            # Not test : split was never used during dev phase
            self.aggr_mlp = SplitMLPs(
                [
                    make_mlp(aggr_mlp_recipe, checkpoint=checkpoint)
                    for _ in aggr_chunk_sizes
                ],
                aggr_chunk_sizes,
            )
            if checkpoint:
                self.aggr_mlp = CheckpointWrapper(self.aggr_mlp)

        self.update_edges = update_edges

    def forward(self, send_rep, rec_rep, edge_rep):
        """
        Apply interaction network to update the representations of receiver nodes,
        and optionally the edge representations.

        send_rep: (N_send, d_h), vector representations of sender nodes
        rec_rep: (N_rec, d_h), vector representations of receiver nodes
        edge_rep: (M, d_h), vector representations of edges used

        Returns:
        rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
        (optionally) edge_rep: (M, d_h), updated vector representations of edges
        """
        # Always concatenate to [rec_nodes, send_nodes] for propagation, but only
        # aggregate to rec_nodes

        node_reps = torch.cat((rec_rep, send_rep), dim=1)
        edge_rep_aggr, edge_diff = self.propagate(
            self.edge_index, x=node_reps, edge_attr=edge_rep
        )

        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connections
        rec_rep = rec_rep + rec_diff

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep

    def message(self, x_j, x_i, edge_attr):
        """
        Compute messages from node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    def aggregate(self, messages, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        aggr = super().aggregate(messages, index, ptr, self.num_rec)
        return aggr, messages


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x):
        """
        Chunk up input and feed through MLPs

        x: (..., N, d), where N = sum(chunk_sizes)

        Returns:
        joined_output: (..., N, d), concatenated results from the different MLPs
        """
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)


def make_mlp(blueprint, layer_norm=True, checkpoint=False):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    mlp = nn.Sequential(*layers)
    if checkpoint:
        mlp = CheckpointWrapper(mlp)
    return mlp
