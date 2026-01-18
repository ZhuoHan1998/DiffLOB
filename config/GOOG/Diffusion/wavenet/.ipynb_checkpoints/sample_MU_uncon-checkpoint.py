class Configuration:

    def __init__(self):

        self.is_training = False
        self.folder_path = "/scratch/prj/genai_techniques_finance/lobsterdata/202302/MU/output-2023-02/0/0/3/"
        self.split_rate = (.85, .05, .10)
        
        self.past_window = 33
        self.predict_window = 32
        self.store_length = 32
        self.n_levels = 10

        self.diff_model = 'wavenet_motion_control'  # ['s4', 'csdi', 'wavenet', 'wavenet_motion', 'wavenet_control', 'wavenet_motion_control']
        self.diff_model_saving_path = 'models/MU/diff_wavenet_vpsde_FTF_MU_AR.pth' # model_architecture_sde_stockname.pth
        
        # training hyperparameters
        self.training_batch_size = 128
        self.n_epochs = 1000
        self.learning_rate = 1e-4
        self.drop_probability = 0.5
        self.weight_decay = 0
        self.clip_gradient = None
        self.ema_rate = 0.999
        self.md_type = "vpsde" 
        self.continuous = False
        self.reduce_mean = True
        self.likelihood_weighting = False
        self.early_stop_patience = 50
        self.early_stop_min_delta = 0.001
        self.motion = False
        self.control = False      
        
        # model
        self.sigma_min = 0.01
        self.sigma_max = 50
        self.beta_min = 0.1
        self.beta_max = 20
        self.num_scales = 100
        
        # sampling hyperparameters
        self.sampling_batch_size = 1
        self.guidance = 0
        self.snr = 0.16
        self.n_steps = 1
        self.probability_flow = False # [True, False]
        self.sampling_noise_removal = True
        self.sampling_method = "pc" # 'pc' or 'ode' or 'dpm_solver'
        self.sampling_predictor = "ancestral_sampling" # ['euler_maruyama', 'reverse_diffusion', 'ancestral_sampling', 'none'] 
        self.sampling_corrector = "none" # ['langevin', 'ald', 'none']
        
        # sampling experiments
        self.AR = True # or False
        self.refresh_cycle = 20
        self.responsive_liquidity = None # ['high', 'low', None]
        self.responsive_imb = None # ['high', 'low', None]
        self.responsive_trend = None # ['high', 'low', None]
        self.responsive_volatility = None # ['high', 'low', None]
        self.liquidity_cond_path = 'conds/MU/liquidity_cond.npy'
        self.imb_cond_path = 'conds/MU/imb_cond.npy'
        self.trend_cond_path = 'conds/MU/trend_cond.npy'
        self.volatility_cond_path = 'conds/MU/volatility_cond.npy'
        self.bin_mode = 'quantile' # ['quantile' or 'range']
        self.samples_saving_path = 'samples/MU/wavenet/vpsde_FTF_AR_uncon.npy' # model_architecture_sde_stockname.pth
        
        
        
        
        