import cv2
import numpy as np
import tensorboard_logger
import os.path as osp
import yaml
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)



class Tb_logger(object):

    def __init__(self):
        pass

    def init_logger(self, path, splits):

        tb_logger = {}
        for split in splits:
            tb_logger[split] = tensorboard_logger.Logger(osp.join(path, split), flush_secs=5, dummy_time=1)

        return tb_logger


class Parser(object):

    def __init__(self, parser):
        parser = self.populate(parser)
        self.opt = parser.parse_args()

    def make_options(self):
        config = yaml.load(open(self.opt.config_path))
        dic = vars(self.opt)
        all(map( dic.pop, config))
        dic.update(config)
        return self.opt


    def populate(self, parser): 
        """ Paths """
        parser.add_argument('--data_path', default='', type=str, help='Data path where to find annotations')
        parser.add_argument('--data_name', default='', type=str, help='Dataset to run on : vrd, unrel')
        parser.add_argument('--logger_dir', default='./runs', type=str, help='Directory to write log and save models')
        parser.add_argument('--thresh_file', default=None ,type=str, help='Specify file for thresholding object detections')
        parser.add_argument('--exp_name', default='' ,type=str, help='Specify name for current experiment if no name given, would generate a random id')
        parser.add_argument('--config_path', default='' ,type=str, help='Path to config file')

        """ Optimization """
        parser.add_argument('--momentum', default=0, type=float, help='Set momentum')
        parser.add_argument('--weight_decay', default=0, type=float, help='Set weight decay')
        parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer to use')
        parser.add_argument('--learning_rate', default=1e-3, type=float, help='Set learning rate')
        parser.add_argument('--lr_update', default=20, type=int, help='Number of iterations before decreasing learning rate')
        parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs for training')
        parser.add_argument('--margin', default=0.2, type=int, help='Set margin for ranking loss')
        parser.add_argument('--use_gpu', default=True, type=bool, help='Whether to run calculations on gpu')
        parser.add_argument('--sampler', default='priority_object', type=str, help='Sampler to use at training')
        parser.add_argument('--start_epoch', default=0, type=int, help='Epoch to start training. Default is 0.')
        parser.add_argument('--save_epoch', default=5, type=int, help='Save model every save_epoch')
        parser.add_argument('--batch_size', default=16, type=int,  help='Batch size')

        """ Inputs to load """
        parser.add_argument('--use_precompappearance', help='whether to use precomputed appearance features', action='store_true')
        parser.add_argument('--use_precompobjectscore', help='whether you use precomputed object scores', action='store_true')
        parser.add_argument('--use_image', help='whether to load image', action='store_true')
        parser.add_argument('--use_ram', help='whether to store features in RAM. Much faster', action='store_true')


        """ Networks to use in each branch """
        parser.add_argument('--net_unigram_s', default='', help='network for unigram subject branch')
        parser.add_argument('--net_unigram_o', default='', help='network for unigram object branch')
        parser.add_argument('--net_unigram_r', default='', help='network unigram predicate branch')
        parser.add_argument('--net_bigram_sr', default='', help='network bigram subject-predicate branch')
        parser.add_argument('--net_bigram_ro', default='', help='network bigram predicate-object branch')
        parser.add_argument('--net_trigram_sro', default='', help='network trigram subject-predicate-object branch')
        parser.add_argument('--net_language', default='', help='language network')
        parser.add_argument('--criterion_name', default='', help='criterion')
        parser.add_argument('--pretrained_model', default='', type=str, help='path to pre-trained visual net with independent classif')
        parser.add_argument('--mixture_keys', default='', type=str, help='keys to use in mixture e.g. s-r-o_sro_sr-r-ro')


        """ Options """
        parser.add_argument('--neg_GT', help='Using negatives candidates', action='store_true')
        parser.add_argument('--sample_negatives', default='among_batch', type=str, help='How to sample negatives when training with embeddings')
        parser.add_argument('--embed_size', default=128, type=int, help='Dimensionality of embedding before classifier')
        parser.add_argument('--d_hidden', default=1024, type=int, help='Dimensionality of hidden layer in projection to joint space')
        parser.add_argument('--num_layers', default=2, type=int, help='Number of projection layers')
        parser.add_argument('--network', default='', type=str, help='Model to use see get_model() from models.py')
        parser.add_argument('--train_split', default='train', type=str, help='train split : either train, trainminusval')
        parser.add_argument('--test_split', default='val', type=str, help='test split : either test, val')
        parser.add_argument('--use_gt', help='whether to use groundtruth objects as candidates', action='store_true')
        parser.add_argument('--add_gt', help='whether to use groundtruth objects as additional candidates during training', action='store_true')
        parser.add_argument('--use_jittering', help='whether to use jittering or not', action='store_true')
        parser.add_argument('--num_negatives', default=3, type=int,  help='Number of negative pairs in a training batch for 1 positive')
        parser.add_argument('--num_workers', default=8, type=int,  help='Number of workers to use. Max is 8.')
        parser.add_argument('--normalize_vis', default=False, type=bool, help='Whether to normalize vis features or not')
        parser.add_argument('--normalize_lang', default=True, type=bool, help='Whether to normalize language features or not')
        parser.add_argument('--scale_criterion', default=1.0, type=float, help='Scaling criterion for log-loss vanishing gradient')
        parser.add_argument('--l2norm_input', help='whether to L2 normalize precomp appearance features and language', action='store_true')
        parser.add_argument('--additional_neg_batch', default=500, type=int, help='Additional negatives to sample in batch')


        """ Evaluation """
        parser.add_argument('--nms_thresh', default=0.5 ,type=float, help='NMS threshold on proposals (used at test time). Candidates are already filtered nms 0.5')
        parser.add_argument('--epoch_model', default='best' ,type=str, help='At which epoch to load the model. Default is best. E.g. epoch50')
        parser.add_argument('--cand_test', default='candidates', type=str, help='Whether to evaluate on GT boxes or candidates')
        parser.add_argument('--subset_test', default='', type=str, help='Which subset to use for evaluation')
        parser.add_argument('--use_objscoreprecomp', help='Use s/o scores from object detector', action='store_true')


        
        """ Test aggreg """
        parser.add_argument('--sim_method', default='emb_word2vec', type=str, help='which similarity method to use')
        parser.add_argument('--thresh_method', default=None, type=str, help='whether to threshold the similarities')
        parser.add_argument('--alpha_r', default=0.5, type=float, help='Weight given to predicate similarity between source and target')
        parser.add_argument('--alpha_s', default=0.0, type=float, help='Weight given to subject similarity between source and target')
        parser.add_argument('--use_target', help='whether to use target triplet as source', action='store_true')
        parser.add_argument('--embedding_type', default='target', type=str, help='Embedding type')
        parser.add_argument('--minimal_mass', default=0, type=float, help='Minimal mass to use')


        """ Analogy """
        parser.add_argument('--use_analogy', help='Whether to use analogy transformation', action='store_true')
        parser.add_argument('--analogy_type', default='hybrid', type=str, help='type of analogy to use')
        parser.add_argument('--gamma', default='deep' ,type=str, help='Which gamma function to use for analogy making. The gamma function computes deformation')
        parser.add_argument('--lambda_reg', default=1, type=int,  help='Weight between regularization term and matching')
        parser.add_argument('--num_source_words_common', default=2, type=int,  help='Minimal number of words in source triplets common to target triplet')
        parser.add_argument('--restrict_source_object', help='Whether to restrict the source triplet to have same object', action='store_true')
        parser.add_argument('--restrict_source_subject', help='Whether to restrict the source triplet to have same subject', action='store_true')
        parser.add_argument('--restrict_source_predicate', help='Whether to restrict the source triplet to have same predicate', action='store_true')
        parser.add_argument('--normalize_source', help='Whether to L2 renormalize the source predictors after aggregation', action='store_true')
        parser.add_argument('--apply_deformation', help='Whether to apply deformation on source triplets', action='store_true')
        parser.add_argument('--precomp_vp_source_embedding', help='Whether to pre-computed vp embedding for source', action='store_true')
        parser.add_argument('--unique_source_random', help='If activated, will sample a unique source triplet at random among pre-selected ones', action='store_true')
        parser.add_argument('--detach_vis', help='Detach target visual visual phrase embedding in analogy branch', action='store_true')
        parser.add_argument('--detach_lang_analogy', help='Whether to detach language embedding in analogy transformation', action='store_true') 


        return parser


    def get_opts_from_dset(self, opt, dset):
        """ Load additional options from dataset object """
        
        opt.vocab_grams     = dset.vocab_grams
        opt.idx_sro_to      = dset.idx_sro_to
        opt.idx_to_vocab    = dset.idx_to_vocab
        opt.word_embeddings = dset.word_embeddings
        opt.d_appearance    = dset.d_appearance
        opt.occurrences     = dset.get_occurrences_precomp(opt.train_split)
        opt.classes         = dset.classes
        opt.predicates      = dset.predicates

        return opt


    def write_opts_dir(self, opt, logger_path):
        """ Write options in directory """

        f = open(osp.join(logger_path, "run_options.yaml"),"w")
        for key, val in vars(opt).iteritems():
            f.write("%s : %s\n" %(key,val))
        f.close()


    def get_res_dir(self, opt, name):
        """ Get results directory : opt.logger_dir/opt.exp_name/name """

        save_dir = osp.join(opt.logger_dir, opt.exp_name, name)

        if 'aggreg' in opt.embedding_type:

            sim_method      = opt.sim_method
            thresh_method   = opt.thresh_method
            use_target      = opt.use_target
            alpha_r         = opt.alpha_r
            alpha_s         = opt.alpha_s
            minimal_mass    = opt.minimal_mass


            sub_dir = 'sim-' + sim_method
            if alpha_r:
                sub_dir = sub_dir + '_' + 'alphar-' + str(alpha_r)
            if alpha_s:
                sub_dir = sub_dir + '_' + 'alphas-' + str(alpha_s)
            if thresh_method:
                sub_dir = sub_dir + '_' + 'tresh-' + thresh_method
            if use_target:
                sub_dir = sub_dir + '_' + 'usetarget'
            if minimal_mass > 0:
                sub_dir = sub_dir + '_' + 'mass-' + str(minimal_mass)

            save_dir = osp.join(save_dir, sub_dir)

        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        return save_dir




