class Configuration:
    def __init__(self):
        # -------- data --------
        self.is_training = False
        self.folder_path = "/scratch/prj/genai_techniques_finance/lobsterdata/202302/MU/output-2023-02/0/0/3/"
        self.split_rate = (.85, .05, .10)

        self.past_window = 33
        self.predict_window = 32
        self.store_length = 32
        self.n_levels = 10

        self.training_batch_size = 128  # not used
        self.sampling_batch_size = 1

        # -------- model --------
        self.d_model = 256
        self.n_layers = 6
        self.state_dim = 64
        self.dropout = 0.0

        # -------- load/save --------
        self.model_saving_path = "models/MU/ar_s5.pth"
        self.samples_saving_path = "samples/MU/ar/ar_s5_low_trend.npy"

        # -------- AR sampling overwrite (same meaning as your other samplers) --------
        self.AR = True
        self.refresh_cycle = 20

        # -------- counterfactual knobs --------
        self.responsive_liquidity = None  # "high"/"low"/None
        self.responsive_imb = None
        self.responsive_trend = "low"
        self.responsive_volatility = None

        self.liquidity_cond_path = "conds/MU/liquidity_cond.npy"
        self.imb_cond_path = "conds/MU/imb_cond.npy"
        self.trend_cond_path = "conds/MU/trend_cond.npy"
        self.volatility_cond_path = "conds/MU/volatility_cond.npy"
        self.bin_mode = "quantile"
