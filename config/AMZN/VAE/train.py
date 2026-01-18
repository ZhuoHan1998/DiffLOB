class Configuration:
    def __init__(self):
        # -------- data --------
        self.is_training = True
        self.folder_path = "/scratch/prj/genai_techniques_finance/lobsterdata/202302/AMZN/output-2023-02/0/0/43/"
        self.split_rate = (.85, .05, .10)

        self.past_window = 33
        self.predict_window = 32
        self.store_length = 32
        self.n_levels = 10

        # dataloader
        self.training_batch_size = 128
        self.sampling_batch_size = 1  # train phase not used

        # -------- training hyperparameters --------
        self.n_epochs = 1000
        self.learning_rate = 2e-4
        self.weight_decay = 0.0
        self.clip_gradient = None
        self.drop_probability = 0.5

        # -------- VAE architecture --------
        self.z_dim = 128
        self.base_channels = 128
        self.cond_channels = 64

        # -------- VAE loss --------
        self.recon_loss_type = "l1"   # "l1" or "mse"
        self.beta_kl = 1.0
        self.kl_anneal = True
        self.kl_anneal_epochs = 300

        # -------- early stopping --------
        self.early_stop_patience = 100
        self.early_stop_min_delta = 1e-4

        # -------- save/load --------
        self.vae_model_saving_path = "models/AMZN/cvae.pth"
        self.vae_samples_saving_path = "samples/AMZN/vae.npy"  # not used in training
