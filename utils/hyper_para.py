from argparse import ArgumentParser


def none_or_default(x, default):
    return x if x is not None else default


class HyperParameters():

    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')

        parser.add_argument('--arch', default='odsnet', type=str, help='model type')
        parser.add_argument('--dataset', default='XRay', choices=['Duke', 'UCL', 'XRay', 'XRayNew'])
        parser.add_argument('--data_root', default='data', type=str, help='dataset dir')
        parser.add_argument('--log_dirs', default='logs', type=str, help='logging dir')
        parser.add_argument('--size', default=[384, 384], help='dataset images original size')
        parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
        parser.add_argument('--lr', default=5e-4, type=float, help='learning rate for training')

        parser.add_argument('--iterations', default=150000, type=int, help='total training iterations')
        parser.add_argument('--finetune', default=10000, type=int, help='')
        parser.add_argument('--steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--start_warm', default=20000, type=int)
        parser.add_argument('--end_warm', default=70000, type=int)

        parser.add_argument('--gamma', default=0.1, type=float, help='LR := LR*gamma at every decay step')
        parser.add_argument('--weight_decay', default=0.05, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--log_text_interval', default=100, type=int)
        # parser.add_argument('--val_interval', default=5, type=int, help='validatioin interval for validation')
        parser.add_argument('--log_image_interval', default=100, type=int)
        parser.add_argument('--save_network_interval', default=25000, type=int)
        parser.add_argument('--save_checkpoint_interval', default=50000, type=int)
        parser.add_argument('--exp_id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')

        # Multi Task
        parser.add_argument('--multitask', action='store_true', help='validatioin interval for validation')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        self.args['down_ratio'] = 4
        self.args['heads'] = None
        if self.args['multitask']:
            heads_dict = {'seg': 1, 'hm': 3, 'dense_bs': 1, 'of': 2}
            self.args['heads'] = none_or_default(self.args['heads'], heads_dict)
        else:
            heads_dict = {'seg': 1}
            self.args['heads'] = none_or_default(self.args['heads'], heads_dict)

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)


class HyperParametersTest():

    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        parser.add_argument('--arch', default='odsnet', type=str, help='model type')
        parser.add_argument('--weights', default='weights/snapshot.pth', help='Path to pretrained network weight only')
        parser.add_argument('--data_root', default='data', help='dataset dir')
        parser.add_argument('--dataset', default='XRay', choices=['Duke', 'UCL', 'XRay', 'XRayNew'])
        parser.add_argument('--size', default=[512, 512], help='dataset images original size')
        parser.add_argument('--split', default='val', help='val/test')
        parser.add_argument('--output', default=None, help='results dir')
        parser.add_argument('--multitask', action='store_true', help='validatioin interval for validation')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        self.args['down_ratio'] = 4
        self.args['heads'] = None
        if self.args['multitask']:
            heads_dict = {'seg': 1, 'hm': 3, 'dense_bs': 1, 'of': 2}
            self.args['heads'] = none_or_default(self.args['heads'], heads_dict)
        else:
            heads_dict = {'seg': 1}
            self.args['heads'] = none_or_default(self.args['heads'], heads_dict)

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
