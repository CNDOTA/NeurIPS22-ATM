import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MemoryAgent, self).__init__()
        self.args = args
        # memory params
        self.mem_slots = args.mem_slots
        self.head_size = args.head_size
        self.num_heads = args.num_heads
        self.mem_size = self.num_heads * self.head_size
        self.num_units = self.args.n_agents + self.args.num_foods
        if self.args.action_mapping:  # duplicate self for (none, n, s, w, e, load) actions
            self.num_units = self.args.n_agents - 1 + self.args.num_foods + self.args.n_actions
        # attention params
        self.k_size = self.head_size  # key size
        self.v_size = self.head_size  # value size
        self.q_size = self.head_size  # query size
        self.qkv_size = self.q_size + self.k_size + self.v_size
        self.total_qkv_size = self.qkv_size * self.num_heads

        # agent obs: food_feats, own_feats, ally_feats, agent_id
        self.food_sta_idx = 0
        self.food_end_idx = self.food_sta_idx + self.args.num_foods * self.args.food_feats_dim
        self.self_sta_idx = self.food_end_idx
        self.self_end_idx = self.self_sta_idx + self.args.unit_feats_dim  # self has 3 features, (x, y, level)
        self.ally_sta_idx = self.self_end_idx
        self.ally_end_idx = self.ally_sta_idx + (self.args.n_agents - 1) * self.args.unit_feats_dim
        self.id_sta_idx = self.ally_end_idx

        # core_feats contains (own_feats, agent_id)
        self.core_dim = self.args.unit_feats_dim + self.args.n_agents

        self.core_emb_layer = nn.Linear(self.core_dim + self.args.n_actions - 1, self.mem_size)  # none with id 0

        # fact_feats contains the same feature list for each unit in (enemy_feats, ally_feats)
        self.food_emb_layer = nn.Linear(3, self.mem_size)
        self.ally_emb_layer = nn.Linear(3, self.mem_size)

        self.mem_emb_layer = nn.Linear(self.mem_size + self.args.mem_slots, self.mem_size)  # add memory age/id
        self.h_layer = nn.Linear(self.mem_size, self.mem_size)

        # self.fc1 = nn.Linear(input_shape, self.mem_size)

        self.q_token = nn.Linear(self.mem_size, self.args.action_tensors)

        # each head has q * k * v linear projector, just using one big param is more efficient
        self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
        self.qkv_layernorm = nn.LayerNorm([self.num_units + self.mem_slots, self.total_qkv_size])

        # used for attend_over_memory function
        # self.fc3 = nn.Linear(self.mem_size, self.mem_size)
        self.fc_mlp = nn.Linear(self.mem_size, self.mem_size)
        self.mlp_memory_layernorm = nn.LayerNorm([self.num_units + self.mem_slots, self.mem_size])
        self.att_memory_layernorm = nn.LayerNorm([self.num_units + self.mem_slots, self.mem_size])

    def init_hidden(self):
        # init memory: we should ensure each row of the memory is initialized to be unique,
        # so initialize the matrix to be the identity. memory: (self.mem_slots, self.mem_size)
        init_state = torch.eye(self.mem_slots, device=self.args.device)
        # pad the matrix with zeros
        if self.mem_size > self.mem_slots:
            difference = self.mem_size - self.mem_slots
            pad = torch.zeros((self.mem_slots, difference), device=self.args.device)
            init_state = torch.cat([init_state, pad], -1)
        # truncation. take the first 'self.mem_size' components
        elif self.mem_size < self.mem_slots:
            init_state = init_state[:, :self.mem_size]
        return init_state.view(1, self.mem_slots * self.mem_size)

    def multihead_attention(self, memory):
        # memory: (batch_size, mem_slot + 1, mem_size)
        qkv = self.qkv_projector(memory)  # (batch_size, 1 + n_enemies + n_allies + num_slot, total_qkv_size)
        # apply layernorm for every dim except the batch dim
        qkv = self.qkv_layernorm(qkv)
        # start multi-head attention
        mem_slots = self.num_units + self.mem_slots  # denoted as N
        # split the qkv to multiple heads H
        # [B, N, F] => [B, N, H, F/H]
        qkv_reshape = qkv.view(-1, mem_slots, self.num_heads, self.qkv_size)  # qkv_size = q_size + k_size + v_size
        # [B, N, H, F/H] => [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)
        # [B, H, N, query_size], [B, H, N, key_size], [B, H, N, value_size]
        q, k, v = torch.split(qkv_transpose, [self.q_size, self.k_size, self.v_size], -1)  # [B, H, N, size]
        # scale q with d_k, the dimensionality of the key vectors
        q = q * (self.k_size ** -0.5)
        # matmul([B, H, N, size], [B, H, size, N]) => [B, H, N, N]
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)
        # matmul([B, H, N, N], [B, H, N, V]) => [B, H, N, V]
        output = torch.matmul(weights, v)
        # [B, H, N, V] => [B, N, H, V] => [B=batch_size * n_agents, N=mem_slots + 1, H*V=mem_size]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0], output_transpose.shape[1], -1))
        return new_memory

    def attend_over_memory(self, memory):
        # memory: (batch_size, mem_slot + 1, mem_size)
        attended_memory = self.multihead_attention(memory)
        # Add a skip connection to the multihead attention memory.
        memory = self.att_memory_layernorm(memory + attended_memory)
        # memory = (memory + attended_memory) / 2
        # add a skip connection to the mlp memory.
        mlp_memory = F.relu(self.fc_mlp(memory))
        memory = self.mlp_memory_layernorm(memory + mlp_memory)
        # memory = (memory + mlp_memory) / 2
        return memory

    def forward(self, inputs, memory):
        batch_size = inputs.shape[0]
        # agent obs: food_feats, own_feats, ally_feats, agent_id
        # core_feats contains (move_feats, own_feats, last_action, agent_id)
        core_feats = torch.cat((inputs[:, self.self_sta_idx: self.self_end_idx], inputs[:, self.id_sta_idx:]), dim=1)
        core_feats = core_feats.reshape(-1, 1, self.core_dim)

        # prepare entities for move and load actions
        core_feats = torch.cat([core_feats] * self.args.n_actions, dim=1)   # (bs*n_agents, 6, core_dim)
        core_id = torch.eye(self.args.n_actions - 1, device=self.args.device)
        core_id = torch.cat([torch.zeros((1, self.args.n_actions - 1), device=self.args.device), core_id], dim=0)  # no-op id keeps 0
        core_id = core_id.expand(batch_size, self.args.n_actions, self.args.n_actions - 1)
        core_plus_id = torch.cat([core_feats, core_id], dim=2)  # (bs*n_agents, 6, core_dim + 6)
        # fact_feats contains (enemy_feats, ally_feats)
        fact_food_feats = inputs[:, self.food_sta_idx: self.food_end_idx]
        fact_food_feats = fact_food_feats.reshape(-1, self.args.num_foods, self.args.food_feats_dim)
        fact_ally_feats = inputs[:, self.ally_sta_idx: self.ally_end_idx]
        fact_ally_feats = fact_ally_feats.reshape(-1, self.args.n_agents - 1, self.args.unit_feats_dim)

        core_emb = self.core_emb_layer(core_plus_id)  # (bs*n_agents, 6, mem_dim)
        fact_food_emb = self.food_emb_layer(fact_food_feats)  # (bs*n_agents, n_enemies, mem_dim)
        fact_ally_emb = self.ally_emb_layer(fact_ally_feats)  # (bs*n_agents, n_allies, mem_dim)

        memory = memory.reshape(-1, self.mem_slots, self.mem_size)  # (batch_size * n_agents, mem_slot, mem_size)
        memory_id = torch.eye(self.mem_slots, device=self.args.device)
        memory_id = memory_id.expand(memory.shape[0], self.mem_slots, self.mem_slots)
        memory_plus_id = torch.cat([memory, memory_id], dim=2)
        mem_emb = self.mem_emb_layer(memory_plus_id)
        # (batch_size * n_agents, 1 + n_enemies + (n_agents - 1) + mem_slot, mem_size)
        memory_plus_factor = torch.cat([core_emb, fact_food_emb, fact_ally_emb, mem_emb], dim=1)
        t_out = self.attend_over_memory(memory_plus_factor)  # (bs*a, num_tokens, mem_size)

        h = t_out[:, :self.args.n_actions, :].mean(dim=1, keepdim=True)  # (batch_size * n_agents, 1, mem_size)

        # h = self.attend_over_memory(memory_plus_factor)[:, :1, :]  # (batch_size * n_agents, 1, mem_size)
        h_mem = torch.tanh(self.h_layer(h))
        next_memory = torch.cat([h_mem, memory[:, :-1, :]], dim=1)  # (batch_size * n_agents, mem_slot, mem_size)

        q_tokens = t_out[:, :self.args.n_actions, :]  # (bs*n_agents, n_actions, mem_size)
        q = self.q_token(q_tokens).mean(dim=2)  # (bs*n_agents, n_actions, n_tensors) => (bs*n_agents, n_actions)

        # q = self.fc2(h)  # (batch_size * n_agents, n_actions)
        return q, next_memory

