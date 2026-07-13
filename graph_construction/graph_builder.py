import pandas as pd
import dgl
import torch
import numpy as np
from typing import Tuple, List, Dict


class FeatureLookup:
    def __init__(self):
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category: str, value):
        self.__category.add(category)

        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value

            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        return self.__inner_bag[value]

    def query_value(self, id: int):
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)

    def get_categories(self):
        return self.__category.copy()

    def get_category_mapping(self, category: str):
        return self.__category_bags.get(category, {}).copy()


def create_graphs_from_context_data(
    client_ctx_df: pd.DataFrame,
    peer_ctx_df: pd.DataFrame,
    client_feature_cols: List[str],
    peer_feature_cols: List[str],
    args,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, FeatureLookup, FeatureLookup]:
    if hasattr(args, "ablation_no_context") and args.ablation_no_context:
        print("Graph construction: context features disabled for ablation")
    else:
        print("Graph construction: context features enabled")

    user_dgl_graph = dgl.graph([])
    serv_dgl_graph = dgl.graph([])
    user_feature_lookup = FeatureLookup()
    serv_feature_lookup = FeatureLookup()

    num_users = len(client_ctx_df)
    for user_id in range(num_users):
        user_feature_lookup.register("User", user_id)

    user_edges_src, user_edges_dst = [], []

    if not (hasattr(args, "ablation_no_context") and args.ablation_no_context):
        for feature_col in client_feature_cols:
            if feature_col in client_ctx_df.columns:
                for value in client_ctx_df[feature_col].unique():
                    user_feature_lookup.register(f"CLIENT_{feature_col.upper()}", value)

        for idx, row in client_ctx_df.iterrows():
            user_node_id = user_feature_lookup.query_id(idx)
            for feature_col in client_feature_cols:
                if feature_col in client_ctx_df.columns:
                    feature_value = row[feature_col]
                    try:
                        feature_node_id = user_feature_lookup.query_id(feature_value)
                        user_edges_src.extend([user_node_id, feature_node_id])
                        user_edges_dst.extend([feature_node_id, user_node_id])
                    except KeyError:
                        print(
                            f"Warning: feature value '{feature_value}' is not registered in the requester graph"
                        )

    user_dgl_graph.add_nodes(len(user_feature_lookup))
    if user_edges_src:
        user_dgl_graph.add_edges(user_edges_src, user_edges_dst)

    num_services = len(peer_ctx_df)
    for serv_id in range(num_services):
        serv_feature_lookup.register("Serv", serv_id)

    serv_edges_src, serv_edges_dst = [], []

    if not (hasattr(args, "ablation_no_context") and args.ablation_no_context):
        for feature_col in peer_feature_cols:
            if feature_col in peer_ctx_df.columns:
                for value in peer_ctx_df[feature_col].unique():
                    serv_feature_lookup.register(f"PEER_{feature_col.upper()}", value)

        for idx, row in peer_ctx_df.iterrows():
            serv_node_id = serv_feature_lookup.query_id(idx)
            for feature_col in peer_feature_cols:
                if feature_col in peer_ctx_df.columns:
                    feature_value = row[feature_col]
                    try:
                        feature_node_id = serv_feature_lookup.query_id(feature_value)
                        serv_edges_src.extend([serv_node_id, feature_node_id])
                        serv_edges_dst.extend([feature_node_id, serv_node_id])
                    except KeyError:
                        print(
                            f"Warning: feature value '{feature_value}' is not registered in the peer graph"
                        )

    serv_dgl_graph.add_nodes(len(serv_feature_lookup))
    if serv_edges_src:
        serv_dgl_graph.add_edges(serv_edges_src, serv_edges_dst)

    user_dgl_graph = dgl.add_self_loop(user_dgl_graph)
    serv_dgl_graph = dgl.add_self_loop(serv_dgl_graph)
    user_dgl_graph = dgl.to_bidirected(user_dgl_graph, copy_ndata=True)
    serv_dgl_graph = dgl.to_bidirected(serv_dgl_graph, copy_ndata=True)

    return user_dgl_graph, serv_dgl_graph, user_feature_lookup, serv_feature_lookup


def create_graphs_from_files(
    client_ctx_file: str,
    peer_ctx_file: str,
    client_feature_cols: List[str] = ["as", "country", "isp", "timezone"],
    peer_feature_cols: List[str] = ["as", "country", "isp", "timezone"],
    args=None,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, FeatureLookup, FeatureLookup]:
    print(f"Loading context data from: {client_ctx_file}, {peer_ctx_file}")

    client_ctx_df = pd.read_csv(client_ctx_file)
    peer_ctx_df = pd.read_csv(peer_ctx_file)

    print(f"Requester context shape: {client_ctx_df.shape}")
    print(f"Peer context shape: {peer_ctx_df.shape}")

    return create_graphs_from_context_data(
        client_ctx_df, peer_ctx_df, client_feature_cols, peer_feature_cols, args
    )


def get_graph_statistics(
    graph: dgl.DGLGraph, feature_lookup: FeatureLookup, graph_name: str = "Graph"
):
    try:
        is_directed = not graph.is_bidirected()
    except AttributeError:
        is_directed = True

    stats = {
        "name": graph_name,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_categories": len(feature_lookup.get_categories()),
        "categories": list(feature_lookup.get_categories()),
        "is_directed": is_directed,
        "has_self_loops": graph.has_self_loop().any().item()
        if graph.number_of_nodes() > 0
        else False,
    }

    if graph.number_of_nodes() > 0:
        in_degrees = graph.in_degrees().float()
        out_degrees = graph.out_degrees().float()

        stats.update(
            {
                "avg_in_degree": in_degrees.mean().item(),
                "avg_out_degree": out_degrees.mean().item(),
                "max_in_degree": in_degrees.max().item(),
                "max_out_degree": out_degrees.max().item(),
                "min_in_degree": in_degrees.min().item(),
                "min_out_degree": out_degrees.min().item(),
            }
        )

    return stats


def print_graph_info(
    user_graph: dgl.DGLGraph,
    serv_graph: dgl.DGLGraph,
    user_lookup: FeatureLookup,
    serv_lookup: FeatureLookup,
):
    print("\n=== Graph construction statistics ===")

    user_stats = get_graph_statistics(user_graph, user_lookup, "Requester graph")
    print(f"\n{user_stats['name']}:")
    print(f"  Nodes: {user_stats['num_nodes']}")
    print(f"  Edges: {user_stats['num_edges']}")
    print(f"  Context categories: {user_stats['num_categories']}")
    print(f"  Categories: {user_stats['categories']}")
    print(f"  Average in-degree: {user_stats.get('avg_in_degree', 0):.2f}")
    print(f"  Average out-degree: {user_stats.get('avg_out_degree', 0):.2f}")

    serv_stats = get_graph_statistics(serv_graph, serv_lookup, "Peer graph")
    print(f"\n{serv_stats['name']}:")
    print(f"  Nodes: {serv_stats['num_nodes']}")
    print(f"  Edges: {serv_stats['num_edges']}")
    print(f"  Context categories: {serv_stats['num_categories']}")
    print(f"  Categories: {serv_stats['categories']}")
    print(f"  Average in-degree: {serv_stats.get('avg_in_degree', 0):.2f}")
    print(f"  Average out-degree: {serv_stats.get('avg_out_degree', 0):.2f}")

    print("\n=== Statistics complete ===\n")
