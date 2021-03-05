class Args():
    def __init__(self):
        self.all_data = True

        # supervised training with absence net
        self.supervised = True
        self.with_absence_net = True

        # unsupervised GAN
        self.pretrained = False

        self.gan = False
        self.wgan = False

        self.reward_net = False
        # self.reward_type = 'valid'  # options: druglikeness, solubility, synthesizability, joint

        self.reward_net_pretrain = False

        self.new_method = True

        self.drugbank = True
        if self.drugbank == True:
            self.max_num_node=100
            self.max_prev_node = self.max_num_node-1
            self.node_feature_dims = 39 #32

        self.QM9_dataset = False
        if self.QM9_dataset ==True:
            self.max_num_node = 100
            self.max_prev_node = self.max_num_node - 1
            self.node_feature_dims = 4