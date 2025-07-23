import argparse
from math import pi as PI

import utils



def train_args_parser():
    parser = argparse.ArgumentParser()

    # DEFAULT SETTINGS
    parser.add_argument("--world_size", help="world size", type=int, default=1) #表示使用的GPU的数量
    parser.add_argument(
        "--distributed", help="distributed", action="store_true", default=False ) #是否使用分布式训练
    parser.add_argument(
        "--autocast", help="autocast", action="store_true", default=False )
    parser.add_argument("--num_workers", help="number of workers", type=int, default=0)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=1)

    # DIRECTORY SETTINGS
    parser.add_argument("--save_dir", help="save directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/code/SAVE_DIR/")
    parser.add_argument("--data_dir", help="data directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/PDBbind_PRO_SAVE_DIR_1/")
    parser.add_argument("--key_dir", help="key directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/keys_1/")

    parser.add_argument("--data_dir_2", help="data directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/PDBbind_PRO_SAVE_DIR_2/")
    parser.add_argument("--key_dir_2", help="key directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/keys_2/")
    parser.add_argument("--restart_dir", help="restart model directory", type=str, default=False)

    # DATASET SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int, default=8)

    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int, default=6)
    parser.add_argument("--num_dense_layers", help="num dense layers", type=int, default=3)
    parser.add_argument(
        "--num_ligand_atom_feature",
        help="ligand atom features",
        type=int,
        default=utils.NUM_LIGAND_ATOM_TYPES,
    )
    parser.add_argument(
        "--num_pocket_atom_feature",
        help="pocket atom features",
        type=int,
        default=utils.NUM_POCKET_ATOM_TYPES,
    )
    parser.add_argument("--num_hidden_feature", help="num hidden features", type=int, default=128) #128
    parser.add_argument("--num_latent_feature", help="num latent features", type=int, default=128)#128
    parser.add_argument("--hidden_feature", help="hidden features", type=int, default=128    )#128
    parser.add_argument("--gamma1", type=float, default=1e1)
    parser.add_argument("--gamma2", type=float, default=5e1)
    parser.add_argument(
        "--dist_one_hot_param1",
        help="dist. one-hot param for representation",
        type=int,
        nargs="+",
        default=[0, 10, 25],
    )
    parser.add_argument(
        "--dist_one_hot_param2",
        help="dist. one-hot param for next distance",
        type=int,
        nargs="+",
        default=[0, 15, 300],
    )
    parser.add_argument("--conditional", help="conditional", action="store_true", default=True)
    parser.add_argument(
        "--num_cond_feature",
        help="num condition features",
        type=int,
        default=utils.NUM_INTERACTION_TYPES,
    )
    parser.add_argument("--ssl", help="semi-supervised learning", action="store_true")

    # TRAINING SETTINGS
    parser.add_argument("--num_epochs", help="num epochs", type=int, default=300) #原来default=1001
    parser.add_argument("--lr", help="lr", type=float, default=1e-3) #1e-3
    parser.add_argument("--lr_decay", help="lr_decay", type=float, default=0.8)
    parser.add_argument("--lr_tolerance", help="lr_tolerance", type=int, default=4)
    parser.add_argument("--lr_min", help="lr_min", type=float, default=1e-6)  #1e-6
    parser.add_argument("--weight_decay", help="weight_decay", type=float, default=1e-5) #0.0
    parser.add_argument(
        "--vae_loss_coeff",
        help="vae coeff for annealing",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
    )
    parser.add_argument(
        "--vae_loss_beta",
        help="decaying coeff for vae loss annealing",
        type=float,
        default=0.2,
    )
    parser.add_argument("--restart_file", help="restart_file", type=str, default=None)
    parser.add_argument("--save_every", help="save every n epochs", type=int, default=1)

    args = parser.parse_args()
    return args

def generate_args_parser():
    parser = argparse.ArgumentParser()

    # DEFAULT SETTINGS
    parser.add_argument("--ngpu", help="ngpu", type=int, default=0)
    parser.add_argument("--ncpu", help="ncpu", type=int, default=30 )  #30
    parser.add_argument("-y", help="delete", action="store_true", default=True)  #y为True才会显示产生过程

    # DIRECTORY SETTINGS
    #####with_ligand#####
    # parser.add_argument("--data_dir", help="data directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/3D-molecular/DeepICL-master/example_with-ligand")
    # parser.add_argument("--key_dir", help="key directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/3D-molecular/DeepICL-master/example_with-ligand")
    # parser.add_argument("--result_dir", help="result directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/3D-molecular/DeepICL-master/example_with-ligand/results_dir")
    ####without_ligand#####
    parser.add_argument("--data_dir", help="data directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/example")
    parser.add_argument("--key_dir", help="key directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/example")
    parser.add_argument("--result_dir", help="result directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/example/results_dir")

    parser.add_argument("--restart_dir", help="restart model directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/code/SAVE_DIR/save_82.pt") #SAVE_DIR_32dim/save_86.pt 85

    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int, default=6)
    parser.add_argument("--num_dense_layers", help="num layers", type=int, default=3)
    parser.add_argument(
        "--num_ligand_atom_feature",
        help="ligand atom features",
        type=int,
        default=utils.NUM_LIGAND_ATOM_TYPES,
    )
    parser.add_argument(
        "--num_pocket_atom_feature",
        help="pocket atom features",
        type=int,
        default=utils.NUM_POCKET_ATOM_TYPES,
    )

    parser.add_argument("--num_hidden_feature", help="num hidden features", type=int, default=128) #default=128
    parser.add_argument("--num_latent_feature", help="num latent features", type=int, default=128)#default=128
    parser.add_argument("--hidden_feature", help="hidden features", type=int, default=128 ) #default=128
    parser.add_argument("--gamma1", type=float, default=1e1)
    parser.add_argument("--gamma2", type=float, default=5e1)

    parser.add_argument(
        "--dist_one_hot_param1",
        help="dist. one-hot param for representation",
        type=int,
        nargs="+",
        default=[0, 10, 25],
    )
    parser.add_argument(
        "--dist_one_hot_param2",
        help="dist. one-hot param for next distance",
        type=int,
        nargs="+",
        default=[0, 15, 300],
    )
    parser.add_argument("--use_scaffold", help="use_scaffold", action="store_true",default=False)  #是否带有配体 
    parser.add_argument("--conditional", help="conditional", action="store_true",default=True)  #是否带有条件
    parser.add_argument("--use_condition", help="use condition", action="store_true",default=True) #action="store_true" 带有ligand和不带有ligand这个条件都是true
    parser.add_argument(
        "--num_cond_feature",
        help="num condition features",
        type=int,
        default=utils.NUM_INTERACTION_TYPES,
    )
    parser.add_argument("--ssl", help="semi-supervised learning", action="store_true")

    # GENERATING SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int, default=8)
    parser.add_argument("--max_num_add_atom", type=int, default=30)
    parser.add_argument("--radial_limits", type=float, nargs="+", default=[0.9, 2.2])  #用于生成网格坐标的径向限制
    parser.add_argument("--num_sample", help="num samples", type=int, default=100)  #默认30
    parser.add_argument("--add_noise", help="add noise", action="store_true", default=True)
    parser.add_argument("--pocket_coeff_max", type=float, default=10.0)
    parser.add_argument("--pocket_coeff_thr", type=float, default=2.5)
    parser.add_argument("--pocket_coeff_beta", type=float, default=0.91)
    parser.add_argument("--dropout", help="dropout parameter", type=float, default=0.0)
    parser.add_argument("--temperature_factor1", type=float, default=0.05) #默认0.1
    parser.add_argument("--temperature_factor2", type=float, default=0.05) #默认0.1
    parser.add_argument("--translation_coeff", type=float, default=0.2)
    parser.add_argument("--rotation_coeff", type=float, default=PI / 90)
    parser.add_argument("--verbose", help="verbal mode", action="store_true", default=True)

    args = parser.parse_args()
    return args


def test_args_parser():
    parser = argparse.ArgumentParser()

    # DEFAULT SETTINGS
    parser.add_argument("--world_size", help="world size", type=int, default=1) #表示使用的GPU的数量
    parser.add_argument(
        "--distributed", help="distributed", action="store_true", default=False ) #是否使用分布式训练
    parser.add_argument(
        "--autocast", help="autocast", action="store_true", default=False )
    parser.add_argument("--num_workers", help="number of workers", type=int, default=0)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=1)

    # DIRECTORY SETTINGS
    parser.add_argument("--save_dir", help="save directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/code/SAVE_DIR/")
    parser.add_argument("--data_dir", help="data directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/PDBbind_PRO_SAVE_DIR_1/")
    parser.add_argument("--key_dir", help="key directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/keys_1/")

    parser.add_argument("--data_dir_2", help="data directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/PDBbind_PRO_SAVE_DIR_2/")
    parser.add_argument("--key_dir_2", help="key directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/data/keys_2/")
    parser.add_argument("--restart_dir", help="restart model directory", type=str, default="/media/estar/98519505-51e9-4e2d-b09d-5a389290bcd9/yh/DeepRL/3D/code/SAVE_DIR/save_18.pt") #save_10.pt不错

    # DATASET SETTINGS
    parser.add_argument("--k", help="k for k-NN parameter", type=int, default=8)

    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int, default=6)
    parser.add_argument("--num_dense_layers", help="num dense layers", type=int, default=3)
    parser.add_argument(
        "--num_ligand_atom_feature",
        help="ligand atom features",
        type=int,
        default=utils.NUM_LIGAND_ATOM_TYPES,
    )
    parser.add_argument(
        "--num_pocket_atom_feature",
        help="pocket atom features",
        type=int,
        default=utils.NUM_POCKET_ATOM_TYPES,
    )
    parser.add_argument("--num_hidden_feature", help="num hidden features", type=int, default=128) #128
    parser.add_argument("--num_latent_feature", help="num latent features", type=int, default=128)#128
    parser.add_argument("--hidden_feature", help="hidden features", type=int, default=128    )#128
    parser.add_argument("--gamma1", type=float, default=1e1)
    parser.add_argument("--gamma2", type=float, default=5e1)
    parser.add_argument(
        "--dist_one_hot_param1",
        help="dist. one-hot param for representation",
        type=int,
        nargs="+",
        default=[0, 10, 25],
    )
    parser.add_argument(
        "--dist_one_hot_param2",
        help="dist. one-hot param for next distance",
        type=int,
        nargs="+",
        default=[0, 15, 300],
    )
    parser.add_argument("--conditional", help="conditional", action="store_true", default=True)
    parser.add_argument(
        "--num_cond_feature",
        help="num condition features",
        type=int,
        default=utils.NUM_INTERACTION_TYPES,
    )
    parser.add_argument("--ssl", help="semi-supervised learning", action="store_true")

    # TRAINING SETTINGS
    parser.add_argument("--num_epochs", help="num epochs", type=int, default=300) #原来default=1001
    parser.add_argument("--lr", help="lr", type=float, default=1e-3) #1e-3
    parser.add_argument("--lr_decay", help="lr_decay", type=float, default=0.8)
    parser.add_argument("--lr_tolerance", help="lr_tolerance", type=int, default=4)
    parser.add_argument("--lr_min", help="lr_min", type=float, default=1e-6)  #1e-6
    parser.add_argument("--weight_decay", help="weight_decay", type=float, default=1e-5) #0.0
    parser.add_argument(
        "--vae_loss_coeff",
        help="vae coeff for annealing",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
    )
    parser.add_argument(
        "--vae_loss_beta",
        help="decaying coeff for vae loss annealing",
        type=float,
        default=0.2,
    )
    parser.add_argument("--restart_file", help="restart_file", type=str, default=None)
    parser.add_argument("--save_every", help="save every n epochs", type=int, default=1)

    args = parser.parse_args()
    return args
