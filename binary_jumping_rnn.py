import torch

class BinaryJumpingRNNCell(torch.nn.Module):
    # expected params:
    # input_dim, hidden_dim, m = 16, r = 2, h_history tensor
    def __init__(self, params):
        super(BinaryJumpingRNNCell, self).__init__()

        self.params = params
        
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.m = params['m']
        self.r = params['r']

        self.W_in = torch.nn.Linear(input_dim, hidden_dim)
        self.W_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_between_layers = torch.nn.Linear(hidden_dim, hidden_dim)

        self.backward_jumps = torch.tensor([1 for _ in range(m)])
        curr_jump = 1
        for i in range(m):
            self.backward_jumps[i] = curr_jump
            curr_jump *= r

        self.h_non_linearity = torch.nn.Sigmoid()
        self.y_non_linearity = torch.nn.Sigmoid()

        self.current_calculating_index = 1
        self.scale_factor = torch.sqrt(torch.tensor(hidden_dim))

        self.params['h_history'][:,0] = 0

    # def get_H(self, batch_size):
    #     indices = torch.full(self.m, self.current_calculating_index)
    #     return self.params['h_history'][:batch_size, indices-self.backward_jumps]

    # inputs (batch, input size)
    # outputs ((batch, hidden size), (batch, hidden size)) [h_i and y_i]
    def forward(self, x):
        batch_size = x.shape[0]

        indices = torch.full(self.m, self.current_calculating_index)
        backward_hops_indices = indices-self.backward_jumps
        backward_hops_indices[backward_hops_indices < 0] = 0
        H = self.params['h_history'][:batch_size, indices-self.backward_jumps]

        K = self.W_k(H)
        q_i = self.W_in(x)
        V = self.W_v(H)

        # is the dim correct?
        softmaxed = torch.nn.Softmax(torch.bmm(torch.transpose(q_i, 1, 2), K)/self.scale_factor, dim = 2)
        softmaxed = softmaxed.repeat(1, self.hidden_dim, 1)

        rs_sum = torch.sum(torch.mul(V, softmaxed), dim = 2)

        h_i = self.h_non_linearity(q_i + rs_sum)

        y_i = self.y_non_linearity(self.W_between_layers(h_i) + h_i)

        if self.current_calculating_index == self.params['h_history'].shape[1]:
            # idk if this resizing is even correct
            self.params['h_history'] = torch.cat((self.params['h_history'],self.params['h_history']), dim=1)
        
        self.params['h_history'][:batch_size, self.current_calculating_index] = h_i
        self.current_calculating_index += 1

        return h_i, y_i


class BinaryJumpingRNN(torch.nn.Module):
    # m is num of back jumps
    # r is exponential growth of back jumps
    def __init__(self, input_dim, hidden_dim, m = 16, r = 2, initial_max_h_history = 2048):
        super(BinaryJumpingRNN, self).__init__()

        self.params = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'm': m,
            'r': r,
            'h_history': torch.zeros(initial_max_h_history, hidden_dim)
        }

        self.cell = BinaryJumpingRNNCell(self.params)

    def add_row_to_history(self, new_h):
        # this is untested
        if self.current_calculating_index >= self.h_history.shape[0]:
            self.h_history = self.h_history.reshape(self.h_history.shape[0] * 2, self.h_history.shape[1])

        self.h_history[self.current_calculating_index] = new_h
        self.current_calculating_index += 1

    # # x.data is a PackedSequence of size (batch_size x num_tokens x d_in)
    # x.data is a PackedSequence of size (total_num_tokens x d_in)
    # x.batch_sizes = num of elements per batches

    # this func should return (total num tokens x d_out)
    def forward(self, x):
        current_index = 0
        max_batch_size = x.batch_sizes[0]

        tmp_output = torch.zeros(x.data.shape[0], self.params['hidden_dim'])

        for parallel_size in x.batch_sizes:
            batched_data = x.data[current_index : current_index + parallel_size]

            h, y = self.cell(batched_data)
            tmp_output[current_index : current_index + parallel_size] = y

            current_index += parallel_size

        return tmp_output