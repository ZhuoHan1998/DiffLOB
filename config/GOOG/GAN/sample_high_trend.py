class Configuration:
    def __init__(self):

        self.is_training = False
        self.folder_path = "/scratch/prj/genai_techniques_finance/lobsterdata/202302/GOOG/output-2023-02/0/0/31/"      
        self.split_rate = (0.85, 0.05, 0.10)

        self.past_window = 33
        self.predict_window = 32
        self.store_length = 32
        self.n_levels = 10
        
        self.gan_samples_saving_path = "samples/GOOG/cgan/cgan_high_trend.npy"

        # model
        self.training_batch_size = 128
        self.n_epochs = 1
        self.early_stop_patience = 100
        self.early_stop_min_delta = 0.001
        self.lr_g = 2e-4
        self.lr_d = 2e-4
        self.weight_decay = 0.0
        self.drop_probability = 0.5
        self.z_dim = 128
        self.base_channels = 128
        self.n_critic = 5
        self.lambda_gp = 10.0
        self.clip_gradient = None

        # sampling hyperparameters
        self.sampling_batch_size = 1
        self.AR = True
        self.refresh_cycle = 20
        self.bin_mode = "quantile"   # quantile or range
        self.responsive_trend = "high"
        self.responsive_volatility = None
        self.responsive_liquidity = None
        self.responsive_imb = None
        self.trend_cond_path = "conds/GOOG/trend_cond.npy"
        self.volatility_cond_path = "conds/GOOG/volatility_cond.npy"
        self.liquidity_cond_path = "conds/GOOG/liquidity_cond.npy"
        self.imb_cond_path = "conds/GOOG/imb_cond.npy"
        
        self.gan_model_saving_path = "models/GOOG/cgan_wgangp.pth"
