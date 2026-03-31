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
        
        Changes on 3/4/26:
        posteriors.vi.diag.build() expects a real Python optimizer object 
        (like torchopt.adam(lr=0.001)) as its optimizer argument, but your yaml 
        file can only store plain values like numbers and strings — so you need to 
        construct the optimizer object in code from the optimizer_lr number before passing it in.
        """
        optimizer_lr = self.build_params.get("optimizer_lr", 0.001) # 3/4/26
        build_kwargs = {
            "optimizer": torchopt.adam(lr=optimizer_lr),
            "init_log_sds": self.build_params["init_log_sds"],
            "temperature": self.build_params["temperature"],
        } # 3/4/26

        # Initiate the VI posterior transform :)
        #self.posterior_transform = posteriors.vi.diag.build(self._log_posterior, **self.build_params) # build_params must be optimizer=torchopt.adam(lr=0.001), init_log_sds=-7, temperature=.1
        self.posterior_transform = posteriors.vi.diag.build(self._log_posterior, **build_kwargs)

        # Initialize the posterior state
        params = dict(self.named_parameters()) # :) Get the model parameters, current weights, as a dict for the posterior transform
        self.posterior_state = self.posterior_transform.init(params) # :) Create starting posterior state, vi_state = vi_transform.init(params) in colab, is the same
    
    # 3/4/26. b start - Added these checkpoint hooks to save and restore the posterior state since Lightning doesn't do that automatically for custom attributes like posterior_state
    def on_save_checkpoint(self, checkpoint):
        if self.posterior_state is not None:
            checkpoint["posterior_params"] = {k: v.detach().cpu() for k, v in self.posterior_state.params.items()}
            checkpoint["posterior_log_sd_diag"] = {k: v.detach().cpu() for k, v in self.posterior_state.log_sd_diag.items()}
    
    def on_load_checkpoint(self, checkpoint):
        # Restore posterior_state when loading from checkpoint - Added by Sam Liechty 3/4/26
        self.init_posterior()
        if "posterior_params" in checkpoint:
            self.posterior_state = self.posterior_state.replace(
                params={k: v for k, v in checkpoint["posterior_params"].items()},
                log_sd_diag={k: v for k, v in checkpoint["posterior_log_sd_diag"].items()}
             )
    # 3/4/26. b end - Added these checkpoint hooks to save and restore the posterior state since Lightning doesn't do that automatically for custom attributes like posterior_state
    
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
    
    
    def _sample_parameters(self, i: int = 0) -> Optional[Dict[str, torch.Tensor]]:
        if self.posterior_state is None:
            raise ValueError("Posterior state is not initialized.")
        params = {}
        for k in self.posterior_state.params.keys():
            mean = self.posterior_state.params[k]
            log_sd = self.posterior_state.log_sd_diag[k]
            params[k] = mean + torch.exp(log_sd) * torch.randn_like(mean)
        return params
    
    def on_fit_start(self):
        # Initialize the posterior state at the start of training
        self.init_posterior()

