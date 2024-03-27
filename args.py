class Args():
    '''
    Macro-parameters of the experiments.
    if supervised == True -> GraphRNN training.
    else -> Graph-Based WGAN training.

    if reward_net == True -> Graph-Based WGAN trained with the support of a RL criterion.
    else -> plain Graph-Based WGAN.

    if reward_net == True, then reward_type defines the metric to optimize.
    options: druglikeness, solubility, synthesizability, joint, fg: functional groupsRL.

    Experiments performed using ZINC data-set.
    '''

    def __init__(self):
        self.max_num_node = 88
        self.edge_feature_dims = 5
        self.max_prev_node = self.max_num_node - 1
        self.node_feature_dims = 12



        # self.ZINC_dataset = True
        # self.ZINC_filtered = True
        # if self.ZINC_dataset == True:
        #     if self.ZINC_filtered:
        #         self.max_num_node = 36
        #         self.edge_feature_dims = 5

        #     else:
        #         self.max_num_node = 36
        #         self.edge_feature_dims = 4


        # # supervised training params
        # self.supervised = True
        # self.test_set = False
