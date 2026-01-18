class Configuration:
    def __init__(self):
        # -------- data --------
        self.is_training = True
        self.folder_path = "/scratch/prj/genai_techniques_finance/lobsterdata/202302/MU/output-2023-02/0/0/3/"
        self.split_rate = (.85, .05, .10)

        self.past_window = 33
        self.predict_window = 32
        self.store_length = 32
        self.n_levels = 10

        self.training_batch_size = 128
        self.sampling_batch_size = 1  # not used in training

        # -------- model --------
        self.d_model = 256
        self.n_layers = 6
        self.state_dim = 64
        self.dropout = 0.0
        
        self.init = "hippo"

        # -------- training --------
        self.n_epochs = 1000
        self.learning_rate = 2e-4
        self.weight_decay = 0.0
        self.clip_gradient = None
        self.recon_loss_type = "l1"   # "l1" or "mse"
        self.drop_probability = 0.5

        # -------- early stopping --------
        self.early_stop_patience = 100
        self.early_stop_min_delta = 1e-4

        # -------- save/load --------
        self.model_saving_path = "models/MU/ar_s5.pth"
        self.samples_saving_path = "samples/MU/ar/ar_s5.npy"
