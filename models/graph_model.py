import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import scipy.sparse as sp
from .attention_layers import SpGraphAttentionLayer
from .dgl_gat_layer import DGLGATLayer
from dgl.nn.pytorch import GraphConv
from efficient_kan import KAN


class SpGAT(torch.nn.Module):
    def __init__(self, graph, nfeat, nhid, dropout, alpha, nheads, args):
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.adj = self.get_adj_nrom_matrix(graph).to(args.devices)
        self.numbers = len(self.adj)
        self.attentions = torch.nn.ModuleList()
        self.nheads = nheads

        for i in range(self.nheads):
            temp = SpGraphAttentionLayer(
                nfeat, nhid, dropout=args.dropout, alpha=alpha, concat=True
            )
            self.attentions += [temp]

        self.dropout_layer = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.out_att = SpGraphAttentionLayer(
            nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, embeds):
        x = self.dropout_layer(embeds)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x = F.elu(self.out_att(x, self.adj))
        return x

    @staticmethod
    def get_adj_nrom_matrix(graph):
        g = graph

        n = g.number_of_nodes()
        in_deg = g.in_degrees().numpy()
        rows = g.edges()[1].numpy()
        cols = g.edges()[0].numpy()
        adj = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))

        def normalize_adj(mx):
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj


class CentralizedGATReliability(torch.nn.Module):
    def __init__(self, args, user_graph, serv_graph, num_users, num_services):
        super(CentralizedGATReliability, self).__init__()
        self.args = args
        self.user_graph = user_graph
        self.serv_graph = serv_graph
        self.num_users = num_users
        self.num_services = num_services
        self.dim = args.dimension

        self.user_embeds = nn.Embedding(user_graph.number_of_nodes(), self.dim)
        nn.init.kaiming_normal_(self.user_embeds.weight)
        self.serv_embeds = nn.Embedding(serv_graph.number_of_nodes(), self.dim)
        nn.init.kaiming_normal_(self.serv_embeds.weight)

        interaction_input_dim_base = self.dim

        if not args.ablation_no_gnn:
            if args.ablation_use_gcn:
                gcn_l1_in, gcn_l1_out = self.dim, 256
                gcn_l2_in, gcn_l2_out = gcn_l1_out, gcn_l1_out

                layer_info = "single-layer GCN" if args.ablation_single_layer else "two-layer GCN"
                print(f"Initializing the {layer_info} backbone")

                self.user_gnn_layer1 = GraphConv(
                    gcn_l1_in, gcn_l1_out, activation=F.silu, allow_zero_in_degree=True
                )
                self.serv_gnn_layer1 = GraphConv(
                    gcn_l1_in, gcn_l1_out, activation=F.silu, allow_zero_in_degree=True
                )
                if not args.ablation_single_layer:
                    self.user_gnn_layer2 = GraphConv(
                        gcn_l2_in, gcn_l2_out, activation=F.silu, allow_zero_in_degree=True
                    )
                    self.serv_gnn_layer2 = GraphConv(
                        gcn_l2_in, gcn_l2_out, activation=F.silu, allow_zero_in_degree=True
                    )

                interaction_input_dim_base = (
                    gcn_l1_out if args.ablation_single_layer else gcn_l2_out
                )

            else:
                gat_l1_in, gat_l1_out_per_head = self.dim, 32
                gat_l2_in = gat_l1_out_per_head * self.args.heads
                gat_l2_out_per_head = gat_l1_out_per_head

                layer_info = "single-layer GAT" if args.ablation_single_layer else "two-layer GAT"
                print(f"Initializing the {layer_info} backbone")

                self.user_gnn_layer1 = DGLGATLayer(
                    gat_l1_in, gat_l1_out_per_head, self.args.heads, self.args.dropout
                )
                self.serv_gnn_layer1 = DGLGATLayer(
                    gat_l1_in, gat_l1_out_per_head, self.args.heads, self.args.dropout
                )
                if not args.ablation_single_layer:
                    self.user_gnn_layer2 = DGLGATLayer(
                        gat_l2_in, gat_l2_out_per_head, self.args.heads, self.args.dropout
                    )
                    self.serv_gnn_layer2 = DGLGATLayer(
                        gat_l2_in, gat_l2_out_per_head, self.args.heads, self.args.dropout
                    )

                interaction_input_dim_base = gat_l2_in
        else:
            print("No GNN backbone; using direct embeddings")

        shared_interaction_input_dim = (
            2 * interaction_input_dim_base if not self.args.inner else interaction_input_dim_base
        )

        if self.args.head_input_norm:
            self.head_input_layernorm = nn.LayerNorm(shared_interaction_input_dim)
        self.head_input_dropout = nn.Dropout(self.args.head_input_dropout)

        sr_output_dim = 1
        if self.args.interaction_type == "mlp":
            print("Initializing MLP for SR prediction head.")
            sr_mlp_layers = []
            current_dim = shared_interaction_input_dim

            for hidden_dim in self.args.sr_head_hidden_dims:
                sr_mlp_layers.append(nn.Linear(current_dim, hidden_dim))
                sr_mlp_layers.append(nn.ReLU())
                current_dim = hidden_dim
            sr_mlp_layers.append(nn.Linear(current_dim, sr_output_dim))
            self.sr_head = nn.Sequential(*sr_mlp_layers)
        elif self.args.interaction_type == "kan":
            kan_sr_layers_spec = (
                [shared_interaction_input_dim] + self.args.sr_head_hidden_dims + [sr_output_dim]
            )
            print(f"Initializing KAN for SR prediction head with layers: {kan_sr_layers_spec}")
            self.sr_head = KAN(
                layers_hidden=kan_sr_layers_spec,
                grid_size=self.args.kan_grid_size,
                spline_order=self.args.kan_spline_order,
            )

        self.other_tasks_head = None
        if self.args.task_mode == "multi":
            other_tasks_output_dim = 3
            if self.args.interaction_type == "mlp":
                print("Initializing MLP for other tasks (RB, RH, RTT) prediction head.")

                others_mlp_layers = []
                current_dim_others = shared_interaction_input_dim
                for hidden_dim_others in self.args.other_tasks_head_hidden_dims:
                    others_mlp_layers.append(nn.Linear(current_dim_others, hidden_dim_others))
                    others_mlp_layers.append(nn.ReLU())
                    current_dim_others = hidden_dim_others
                others_mlp_layers.append(nn.Linear(current_dim_others, other_tasks_output_dim))
                self.other_tasks_head = nn.Sequential(*others_mlp_layers)
            elif self.args.interaction_type == "kan":
                kan_others_layers_spec = (
                    [shared_interaction_input_dim]
                    + self.args.other_tasks_head_hidden_dims
                    + [other_tasks_output_dim]
                )
                print(
                    f"Initializing KAN for other tasks (RB, RH, RTT) head with layers: {kan_others_layers_spec}"
                )
                self.other_tasks_head = KAN(
                    layers_hidden=kan_others_layers_spec,
                    grid_size=self.args.kan_grid_size,
                    spline_order=self.args.kan_spline_order,
                )

        self.cache = {}

    def forward(self, userIdx, servIdx, train_flag):
        batch_user_reps, batch_serv_reps = None, None

        if train_flag:
            if not self.args.ablation_no_gnn:
                user_embeds = self.user_embeds(
                    torch.arange(self.user_graph.number_of_nodes()).to(self.args.devices)
                )
                serv_embeds = self.serv_embeds(
                    torch.arange(self.serv_graph.number_of_nodes()).to(self.args.devices)
                )

                user_h_l1 = self.user_gnn_layer1(self.user_graph, user_embeds)
                serv_h_l1 = self.serv_gnn_layer1(self.serv_graph, serv_embeds)

                if not self.args.ablation_single_layer:
                    user_h_final = self.user_gnn_layer2(self.user_graph, user_h_l1)
                    serv_h_final = self.serv_gnn_layer2(self.serv_graph, serv_h_l1)
                else:
                    user_h_final = user_h_l1
                    serv_h_final = serv_h_l1

                batch_user_reps = user_h_final[userIdx]
                batch_serv_reps = serv_h_final[servIdx]
            else:
                batch_user_reps = self.user_embeds(userIdx)
                batch_serv_reps = self.serv_embeds(servIdx)
        else:
            batch_user_reps = self.cache["user"][userIdx]
            batch_serv_reps = self.cache["serv"][servIdx]

        if not self.args.inner:
            shared_interaction_input_features = torch.cat(
                (batch_user_reps, batch_serv_reps), dim=-1
            )
        else:
            shared_interaction_input_features = batch_user_reps * batch_serv_reps

        if hasattr(self, "head_input_layernorm"):
            module_input = self.head_input_layernorm(shared_interaction_input_features)
        else:
            module_input = shared_interaction_input_features

        module_input = self.head_input_dropout(module_input)

        if self.args.task_mode == "single":
            if not hasattr(self, "sr_head"):
                raise AttributeError(
                    "sr_head is not initialized. Check model __init__ for single task mode."
                )
            raw_predictions = self.sr_head(module_input)

        elif self.args.task_mode == "multi":
            if not (
                hasattr(self, "sr_head")
                and hasattr(self, "other_tasks_head")
                and self.other_tasks_head is not None
            ):
                raise AttributeError(
                    "sr_head or other_tasks_head is not initialized. Check model __init__ for multi task mode."
                )

            raw_sr_pred = self.sr_head(module_input)
            raw_others_pred = self.other_tasks_head(module_input)

            raw_predictions = torch.cat((raw_sr_pred, raw_others_pred), dim=1)
        else:
            raise ValueError(f"Unsupported task_mode: {self.args.task_mode}")

        predicted_values = raw_predictions.sigmoid()

        if self.args.task_mode == "single":
            if predicted_values.ndim > 1 and predicted_values.shape[1] == 1:
                predicted_values = predicted_values.squeeze(-1)

        return predicted_values

    def prepare_test_model(self):
        with torch.no_grad():
            if not self.args.ablation_no_gnn:
                user_embeds = self.user_embeds(
                    torch.arange(self.user_graph.number_of_nodes()).to(self.args.devices)
                )
                serv_embeds = self.serv_embeds(
                    torch.arange(self.serv_graph.number_of_nodes()).to(self.args.devices)
                )

                user_h_l1 = self.user_gnn_layer1(self.user_graph, user_embeds)
                serv_h_l1 = self.serv_gnn_layer1(self.serv_graph, serv_embeds)

                if not self.args.ablation_single_layer:
                    user_h_final = self.user_gnn_layer2(self.user_graph, user_h_l1)
                    serv_h_final = self.serv_gnn_layer2(self.serv_graph, serv_h_l1)
                else:
                    user_h_final = user_h_l1
                    serv_h_final = serv_h_l1

                self.cache["user"] = user_h_final[
                    torch.arange(self.num_users).to(self.args.devices)
                ]
                self.cache["serv"] = serv_h_final[
                    torch.arange(self.num_services).to(self.args.devices)
                ]
            else:
                user_indices = torch.arange(self.num_users).to(self.args.devices)
                serv_indices = torch.arange(self.num_services).to(self.args.devices)
                self.cache["user"] = self.user_embeds(user_indices)
                self.cache["serv"] = self.serv_embeds(serv_indices)

    def get_embeds_parameters(self):
        parameters = []
        for params in self.user_embeds.parameters():
            parameters.append(params)
        for params in self.serv_embeds.parameters():
            parameters.append(params)
        return parameters

    def get_attention_parameters(self):
        if self.args.ablation_no_gnn:
            return []

        parameters = []
        if hasattr(self, "user_gnn_layer1"):
            parameters.extend(list(self.user_gnn_layer1.parameters()))
        if hasattr(self, "serv_gnn_layer1"):
            parameters.extend(list(self.serv_gnn_layer1.parameters()))

        if not self.args.ablation_single_layer:
            if hasattr(self, "user_gnn_layer2"):
                parameters.extend(list(self.user_gnn_layer2.parameters()))
            if hasattr(self, "serv_gnn_layer2"):
                parameters.extend(list(self.serv_gnn_layer2.parameters()))

        return parameters

    def get_mlp_parameters(self):
        parameters = []

        if hasattr(self, "sr_head") and self.sr_head is not None:
            parameters.extend(list(self.sr_head.parameters()))

        if self.args.task_mode == "multi":
            if hasattr(self, "other_tasks_head") and self.other_tasks_head is not None:
                parameters.extend(list(self.other_tasks_head.parameters()))

        return parameters

    def get_last_shared_layer_parameters(self):
        if self.args.ablation_no_gnn:
            return []

        parameters = []
        if self.args.ablation_single_layer:
            if hasattr(self, "user_gnn_layer1"):
                parameters.extend(list(self.user_gnn_layer1.parameters()))
            if hasattr(self, "serv_gnn_layer1"):
                parameters.extend(list(self.serv_gnn_layer1.parameters()))
        else:
            if hasattr(self, "user_gnn_layer2"):
                parameters.extend(list(self.user_gnn_layer2.parameters()))
            if hasattr(self, "serv_gnn_layer2"):
                parameters.extend(list(self.serv_gnn_layer2.parameters()))

        return parameters
