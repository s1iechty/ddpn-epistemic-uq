from typing import Any, Dict, Optional
import torch
import posteriors
from deep_uncertainty.training.losses import double_poisson_nll as nll
from deep_uncertainty.models.bayesian_uq.bayesian_nn import DoublePoissonBayesianNN
import torchopt
# Pull Request into Ryan Vance's GitHub repository when done

class VI_DoublePoissonNN(DoublePoissonBayesianNN):

    def __init__(self, n_data, build_params, **kwargs):
        self.n_data = n_data
        self.build_params = build_params
        # Initialize the base class
        super().__init__(**kwargs)
        
        # Initialize state container
        self.posterior_state = None


    def _log_posterior(self, params, batch):
        x_batch, y_batch = batch
        output = self.functional(params, x_batch)

        log_lik = - nll(output, y_batch) * self.n_data / len(x_batch)
        log_prior = posteriors.utils.diag_normal_log_prob(params, mean=0.0, sd_diag=0.5)
        return log_lik + log_prior, torch.tensor([])


    def init_posterior(self):
        """
        Subclasses must define the 'posteriors' transform (VI, Laplace, etc.)
        and initialize self.posterior_state here.
        """
        # Initiate the VI posterior transform
        self.posterior_transform = posteriors.vi.diag.build(self._log_posterior, **self.build_params)
        
        # Initialize the posterior state
        params = dict(self.named_parameters())
        self.posterior_state = self.posterior_transform.init(params)
        

    def training_step(self, batch: Any, batch_idx: int):

        # 1. Initalize the posterior state if it hasn't been already
        if self.posterior_state is None:
            self.init_posterior()

        # 2. Update the posterior state using the current batch
        self.posterior_state, stats = self.posterior_transform.update(
            self.posterior_state, 
            batch
        )

        # 3. Log Loss (Negative ELBO)
        self.log("train_nelbo", self.posterior_state.nelbo, prog_bar=True)

        return None
    
    
    def _sample_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        # For VI, we sample from the variational distribution defined by the current posterior state.
        if self.posterior_state is None:
            return None
        # Draw one set of weights from the variational posterior distribution
        return posteriors.vi.diag.sample(self.posterior_state)