from typing import Any, Dict, Optional
import torch
import posteriors
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.training.losses import double_poisson_nll as nll
from deep_uncertainty.models.bayesian_uq.bayesian_nn import DoublePoissonBayesianNN
import torchopt
from deep_uncertainty.models.backbones import MLP

# cs-vpn.byu.edu IP address for Global VPN to access Trinity
# :) means I checked it against SGMCMC and Code in Colab and Claude and it checks out, SO WHY AM I BUGGING OUT???

class VI_DoublePoissonNN(DoublePoissonBayesianNN): # :)

    def __init__(self, n_data, build_params, **kwargs): # maybe I make individual params instead of build_params  
        super().__init__(**kwargs) # :)

        # Init subclass specific
        self.n_data = n_data # :)
        self.build_params = build_params # :)

        # Initialize state container
        self.posterior_state = None # :)


    def _log_posterior(self, params, batch): # :)
        x_batch, y_batch = batch
        output = self.functional(params, x_batch) # :)

        log_lik = - nll(output, y_batch) * self.n_data / len(x_batch) # :)
        log_prior = posteriors.utils.diag_normal_log_prob(params, mean=0.0, sd_diag=0.5) # :)
        return log_lik + log_prior, torch.tensor([]) # :)  


    def init_posterior(self): # :)
        """
        Subclasses must define the 'posteriors' transform (VI, Laplace, etc.)
        and initialize self.posterior_state here.
        """

        # Initiate the VI posterior transform :)
        self.posterior_transform = posteriors.vi.diag.build(self._log_posterior, **self.build_params) # build_params must be optimizer=torchopt.adam(lr=0.001), init_log_sds=-7, temperature=.1
        
        # Initialize the posterior state
        params = dict(self.named_parameters()) # :) Get the model parameters, current weights, as a dict for the posterior transform
        self.posterior_state = self.posterior_transform.init(params) # :) Create starting posterior state, vi_state = vi_transform.init(params) in colab, is the same
        

    # :)
    def training_step(self, batch: Any): #, batch_idx: int): 

        # 1. Initalize the posterior state if it hasn't been already
        if self.posterior_state is None:
            self.init_posterior()

        # 2. Update the posterior state using the current batch       
        self.posterior_state, stats = self.posterior_transform.update(  # Exactly what I have in COLAB
            self.posterior_state,   
            batch                                                       # Except (x_batch, y_batch) is just batch here since we unpack in the _log_posterior function
        )

        # 3. Log Loss (Negative ELBO)
        self.log("train_nelbo", self.posterior_state.nelbo, prog_bar=True)

        return None
    
    
    def _sample_parameters(self) -> Optional[Dict[str, torch.Tensor]]: # :)
        # For VI, we sample from the variational distribution defined by the current posterior state.
        if self.posterior_state is None: # :)
            return None
        # Draw one set of weights from the variational posterior distribution
        return posteriors.vi.diag.sample(self.posterior_state) # :) Same as Colab
    
    # Training, train model takes in config file, Deep Uncertainy -> evaluation -> get train_model.py (READ ME HAS INSTRUCTIONS ON HOW TO DO THIS) specify chekpoint outpoint folder eval_model.py and then training, want a training config file  

# TESTING VI
# "python -m deep_uncertainty.models.bayesian_uq.vi_nn" to run the code
if __name__ == "__main__":
    print("Script Started")
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    
    # -----------------------------------
    # 1. Generate toy data
    # -----------------------------------

    torch.manual_seed(42)

    n_points = 400

    x_all = torch.rand(n_points) * 10 - 3  # range [-3, 7]


    mask_outside = (x_all < 0) | (x_all > 2)
    x_train_outside = x_all[mask_outside]

    mask_inside = (x_all >= 0) & (x_all <= 2)
    x_train_inside = x_all[mask_inside][::5]


    x_train = torch.cat([x_train_outside, x_train_inside]).unsqueeze(1)

    lambda_x_all = torch.exp(0.7 * x_all - 0.05 * x_all**2 + 1.0)
    lambda_x_train = torch.exp(0.7 * x_train - 0.05 * x_train**2 + 1.0)

    y_all = torch.poisson(lambda_x_all)
    y_train = torch.poisson(lambda_x_train)


    fig, axes = plt.subplots(1,2, figsize=(14, 4))
    axes[0].scatter(x_train.numpy(), y_train.numpy(), alpha=0.6)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('count y')
    axes[0].set_title('Training Data')

    axes[1].scatter(x_all.numpy(), y_all.numpy(), alpha=0.6)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('count')
    axes[1].set_title('All Data')
    # Use blocking show so the window remains until the user closes it
    plt.show()

    # -----------------------------------
    # 2. Create DataLoader
    # -----------------------------------

    dataset = TensorDataset(x_train, y_train)
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("DataLoader created with batch size:", batch_size)

    # -----------------------------------
    # 3. Instantiate the model
    # -----------------------------------
    
    build_params = {
        "optimizer": torchopt.adam(lr=0.001),
        "init_log_sds": -7.0,
        "temperature": 0.1
    }

    model = VI_DoublePoissonNN(
        n_data=len(x_train),
        build_params=build_params,
        # passed up to DoublePoissonNN via **kwargs
        backbone_type=MLP,
        backbone_kwargs={"input_dim": 1, "output_dim": 64},
        optim_type=OptimizerType.ADAM,
        optim_kwargs={"lr": 0.001},
        )
    
    print("Model instantiated with VI posterior.")
   
    # Ok I pray to God that that worked, now the next step

    #------------------------------------
    # 4. How to Train Your Model
    #------------------------------------
    from tqdm import tqdm

    model.train()
    nelbos = []

    for epoch in tqdm(range(400)):
        for batch in loader:
            model.training_step(batch)
            nelbos.append(model.posterior_state.nelbo.item())

    print("Training complete")

    # Plot the NELBO to check it's decreasing
    import matplotlib.pyplot as plt
    plt.plot(nelbos)
    plt.xlabel("Step")
    plt.ylabel("Negative ELBO")
    plt.title("VI Training Loss")
    plt.show()