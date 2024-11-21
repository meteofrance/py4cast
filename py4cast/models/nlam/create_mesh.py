import matplotlib
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx

from py4cast.utils import torch_save


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting. This function fixes this
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def plot_graph(graph, title=None):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)
    # TODO: indicate direction of directed edges

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def mk_2d_graph(xy, nx, ny):
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])
    if xm == xM or ym == yM:
        raise ValueError("Your graph is constant either on x dimension or y dimension.")

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(ly), len(lx))

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([mg[0][node], mg[1][node]])

    # add diagonal edges
    g.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
        + [((x + 1, y), (x, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
    )

    # turn into directed graph
    dg = networkx.DiGraph(g)
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    return dg


def save_edges(graph, name, base_path):
    torch_save(graph.edge_index, base_path / f"{name}_edge_index.pt")

    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        torch.float32
    )  # Save as float32
    torch_save(edge_features, base_path / f"{name}_features.pt")


def save_edges_list(graphs, name, base_path):
    list_edge_index = [graph.edge_index for graph in graphs]
    torch_save(list_edge_index, base_path / f"{name}_edge_index.pt")
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(torch.float32)
        for graph in graphs
    ]  # Save as float32
    torch_save(edge_features, base_path / f"{name}_features.pt")


########################################################################################


def hierarchical_mesh(G, mesh_levels, plot, cache_dir_path):
    # Relabel nodes of each level with level index first
    G = [prepend_node_index(graph, level_i) for level_i, graph in enumerate(G)]

    num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
    # First node index in each level in the hierarcical graph
    first_index_level = np.concatenate(
        (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
    )

    # Create inter-level mesh edges
    mesh_up = []
    mesh_down = []
    for from_level, to_level, G_from, G_to, start_index in zip(
        range(1, mesh_levels),
        range(0, mesh_levels - 1),
        G[1:],
        G[:-1],
        first_index_level[: mesh_levels - 1],
    ):
        # start out from graph at from level
        G_down = G_from.copy()
        G_down.clear_edges()
        G_down = networkx.DiGraph(G_down)

        # Add nodes of to level
        G_down.add_nodes_from(G_to.nodes(data=True))

        # build kd tree for mesh point pos
        # order in vm should be same as in vm_xy
        v_to_list = list(G_to.nodes)
        v_from_list = list(G_from.nodes)
        v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
        kdt_m = scipy.spatial.KDTree(v_from_xy)

        # add edges from mesh to grid
        for v in v_to_list:
            # find 1(?) nearest neighbours (index to vm_xy)
            neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
            u = v_from_list[neigh_idx]

            # add edge from mesh to grid
            G_down.add_edge(u, v)
            d = np.sqrt(np.sum((G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2))
            G_down.edges[u, v]["len"] = d
            G_down.edges[u, v]["vdiff"] = (
                G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
            )

        # relabel nodes to integers (sorted)
        G_down_int = networkx.convert_node_labels_to_integers(
            G_down, first_label=start_index, ordering="sorted"
        )  # Issue with sorting here
        G_down_int = sort_nodes_internally(G_down_int)
        pyg_down = from_networkx_with_start_index(G_down_int, start_index)

        # Create up graph, invert downwards edges
        up_edges = torch.stack((pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0)
        pyg_up = pyg_down.clone()
        pyg_up.edge_index = up_edges

        mesh_up.append(pyg_up)
        mesh_down.append(pyg_down)

        if plot:
            plot_graph(pyg_down, title=f"Down graph, {from_level} -> {to_level}")
            plt.show()

            plot_graph(pyg_down, title=f"Up graph, {to_level} -> {from_level}")
            plt.show()

    # Save up and down edges
    save_edges_list(mesh_up, "mesh_up", cache_dir_path)
    save_edges_list(mesh_down, "mesh_down", cache_dir_path)

    # Extract intra-level edges for m2m
    m2m_graphs = [
        from_networkx_with_start_index(
            networkx.convert_node_labels_to_integers(
                level_graph, first_label=start_index, ordering="sorted"
            ),
            start_index,
        )
        for level_graph, start_index in zip(G, first_index_level)
    ]

    mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

    # For use in g2m and m2g
    G_bottom_mesh = G[0]

    joint_mesh_graph = networkx.union_all([graph for graph in G])
    all_mesh_nodes = joint_mesh_graph.nodes(data=True)

    return m2m_graphs, mesh_pos, G_bottom_mesh, all_mesh_nodes


########################################################################################


def monolevel_mesh(G, nx, plot):
    # combine all levels to one graph
    G_tot = G[0]
    for lev in range(1, len(G)):
        nodes = list(G[lev - 1].nodes)
        n = int(np.sqrt(len(nodes)))
        ij = (
            np.array(nodes)
            .reshape((n, n, 2))[1::nx, 1::nx, :]
            .reshape(int(n / nx) ** 2, 2)
        )
        ij = [tuple(x) for x in ij]
        G[lev] = networkx.relabel_nodes(G[lev], dict(zip(G[lev].nodes, ij)))
        G_tot = networkx.compose(G_tot, G[lev])

    # Relabel mesh nodes to start with 0
    G_tot = prepend_node_index(G_tot, 0)

    # relabel nodes to integers (sorted)
    G_int = networkx.convert_node_labels_to_integers(
        G_tot, first_label=0, ordering="sorted"
    )

    # Graph to use in g2m and m2g
    G_bottom_mesh = G_tot
    all_mesh_nodes = G_tot.nodes(data=True)

    # export the nx graph to PyTorch geometric
    pyg_m2m = from_networkx(G_int)
    m2m_graphs = [pyg_m2m]
    mesh_pos = [pyg_m2m.pos.to(torch.float32)]

    if plot:
        plot_graph(pyg_m2m, title="Mesh-to-mesh")
        plt.show()

    return m2m_graphs, mesh_pos, G_bottom_mesh, all_mesh_nodes


########################################################################################


def build_graph_for_grid(
    grid_xy: torch.Tensor,
    cache_dir_path: str,
    plot: bool = False,
    levels: int = None,
    hierarchical: bool = False,
):
    """
    xy: (x, y) coordinates of grid points, shape (2, N_x, N_y).
    plot: If graphs should be plotted during generation (default: False).
    levels: Limit multi-scale mesh to given number of levels, from bottom up (default: None (no limit)).
    hierarchical: Generate hierarchical mesh graph (default: 0, no).
    """

    xy = grid_xy.numpy()
    grid_xy = torch.tensor(xy)
    pos_max = torch.max(torch.abs(grid_xy))

    # graph geometry
    nx = 3  # number of children = nx**2
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2

    mesh_levels = nlev - 1
    if levels:
        # Limit the levels in mesh graph
        mesh_levels = min(mesh_levels, levels)

    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # multi resolution tree levels
    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = mk_2d_graph(xy, n, n)
        if plot:
            plot_graph(from_networkx(g), title=f"Mesh graph, level {lev}")
            plt.show()

        G.append(g)

    if hierarchical:
        m2m_graphs, mesh_features, G_bottom_mesh, all_mesh_nodes = hierarchical_mesh(
            G, mesh_levels, plot, cache_dir_path
        )

    else:
        m2m_graphs, mesh_features, G_bottom_mesh, all_mesh_nodes = monolevel_mesh(
            G, nx, plot
        )

    # Save m2m edges
    save_edges_list(m2m_graphs, "m2m", cache_dir_path)

    # Divide mesh node pos by max coordinate of grid cell
    mesh_features = [pos / pos_max for pos in mesh_features]

    print("In create grid_mesh mesh_pos", mesh_features[0].shape)

    # Save mesh positions
    torch_save(
        mesh_features, cache_dir_path / "mesh_features.pt"
    )  # mesh pos, in float32

    G_g2m, vm, vm_xy, vg_list = grid2mesh(
        G_bottom_mesh, all_mesh_nodes, xy, plot, cache_dir_path
    )

    mesh2grid(G_g2m, vm, vm_xy, vg_list, plot, cache_dir_path)


########################################################################################


def grid2mesh(G_bottom_mesh, all_mesh_nodes, xy, plot, cache_dir_path):
    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    print("Bottom_mesh nodes", len(vm))
    vm_xy = np.array([xy for _, xy in vm.data("pos")])
    # distance between mesh nodes
    dm = np.sqrt(np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2))

    print(f"Distance between mesh node {dm}, {DM_SCALE}")
    # grid nodes
    Ny, Nx = xy.shape[1:]

    print(f"In create Mesh Ny and Nx shape {Ny}, {Nx}")

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    for node in G_grid.nodes:
        # pos is in feature but here explicit for convenience
        G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])

    print(f"Len G_grid {len(G_grid.nodes)}")
    # add 1000 to node key to separate grid nodes (1000,i,j) from mesh nodes (i,j)
    # and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, 1000)

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # now add (all) mesh nodes, include features (pos)
    G_grid.add_nodes_from(all_mesh_nodes)

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges
    for v in vm:
        # find neighbours (index to vg_xy)
        neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
        for i in neigh_idxs:
            u = vg_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2))
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]

    pyg_g2m = from_networkx(G_g2m)

    if plot:
        plot_graph(pyg_g2m, title="Grid-to-mesh")
        plt.show()

    save_edges(pyg_g2m, "g2m", cache_dir_path)

    return G_g2m, vm, vm_xy, vg_list


########################################################################################


def mesh2grid(G_g2m, vm, vm_xy, vg_list, plot, cache_dir_path):
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    vm_list = list(vm)
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # add edges from mesh to grid
    for v in vg_list:
        # find 4 nearest neighbours (index to vm_xy)
        neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 4)[1]
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            d = np.sqrt(np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2))
            G_m2g.edges[u, v]["len"] = d
            G_m2g.edges[u, v]["vdiff"] = G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]

    # relabel nodes to integers (sorted)
    G_m2g_int = networkx.convert_node_labels_to_integers(
        G_m2g, first_label=0, ordering="sorted"
    )
    pyg_m2g = from_networkx(G_m2g_int)

    if plot:
        plot_graph(pyg_m2g, title="Mesh-to-grid")
        plt.show()

    # m2g
    save_edges(pyg_m2g, "m2g", cache_dir_path)
