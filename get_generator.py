from model import GRU_plain
from args import Args


def get_generator():
    args = Args()

    num_layers = 4

    # NODE LEVEL AND ABSENCE NET
    embedding_size_rnn = 64
    hidden_size_rnn = 64

    # EDGE LEVEL
    embedding_size_rnn_output = 32
    hidden_size_rnn_output = 32
    out_edge_level = hidden_size_rnn_output // 2

    rnn = GRU_plain(input_size=args.node_feature_dims+5*args.max_prev_node, embedding_size=embedding_size_rnn,
                    hidden_size=hidden_size_rnn, num_layers=num_layers, has_input=True,
                    has_output=True, output_size=hidden_size_rnn_output, node_lvl=True,
                    out_middle_layer=hidden_size_rnn_output)

    if (args.with_absence_net == True):
        absence_net = GRU_plain(input_size=5, embedding_size=embedding_size_rnn_output,
                                hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                                has_output=True, output_size=1, node_lvl=False, out_middle_layer=out_edge_level)

        output = GRU_plain(input_size=5, embedding_size=embedding_size_rnn_output,
                           hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                           has_output=True, output_size=4, node_lvl=False,
                           out_middle_layer=out_edge_level)  # output_size=4

    else:
        output = GRU_plain(input_size=5, embedding_size=embedding_size_rnn_output,
                           hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
                           has_output=True, output_size=4, node_lvl=False,
                           out_middle_layer=out_edge_level)  # output_size=4

    if (args.with_absence_net == True):
        return rnn, output, absence_net
    else:
        return rnn, output
