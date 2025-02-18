import torch
import torch.nn as nn
from adabmDCA.stats import get_freq_single_point, get_freq_two_points
from tqdm import tqdm


class plmDCA(nn.Module):
    def __init__(
        self,
        L: int,
        q: int,
    ):
        """Initializes the plmDCA model.

        Args:
            L (int): Number of residues in the MSA.
            q (int): Number of states for the categorical variables.

        """
        super(plmDCA, self).__init__()
        self.L = L
        self.q = q
        self.h = nn.Parameter(torch.randn(L, q))
        self.J = nn.Parameter(torch.randn(self.L, self.q, self.L, self.q))
        self.mask = nn.Parameter(torch.ones_like(self.J), requires_grad=False)
        for i in range(self.L):
            self.mask.data[i, :, i, :] = 0

    def forward(
        self,
        X: torch.Tensor,
        residue_idxs: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Performes a sweep over the MSA to try to change the residues.
        
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            residue_idxs (torch.Tensor): Soring of the residues to update.
            beta (float, optional): Inverse temperature. Defaults to 1.0.
        
        Returns:
            torch.Tensor: Updated MSA with the mutated residues.
        """
        N, L, q = X.shape
        X_flat = X.view(N, -1)
        J_flat = self.J.reshape(L, q, L * q)
        for i in residue_idxs:
            logits_i = self.h[i].unsqueeze(0) + (X_flat @ J_flat[i].mT)
            prob_i = nn.functional.softmax(beta * logits_i, dim=-1)
            X[:, i] = torch.nn.functional.one_hot(torch.multinomial(prob_i, 1), num_classes=q).to(X.dtype).squeeze(1)
        
        return X
        
        
    def remove_autocorr(self):
        """Removes the autocorrelation from the model parameters."""
        self.J.data = self.J.data * self.mask.data
        
        
    def prob_i(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the one-site cavity probabilities of the model.

        Args:
            X (torch.Tensor): Input MSA one-hot encoded.

        Returns:
            torch.Tensor: One-site cavity probabilities.
        """
        logit = self.h + torch.einsum("njb, iajb -> nia", X, self.J)
        prob = nn.functional.softmax(logit, dim=-1)
        
        return prob
        
        
    def compute_gradient(
        self,
        X: torch.Tensor,
        fi_target: torch.Tensor,
        fij_target: torch.Tensor,
        weights: torch.Tensor,
        reg_h: float = 0.0,
        reg_J: float = 0.0,
    ) -> None:
        """Computes the gradient of the model's parameters.
        
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            fi_target (torch.Tensor): Single-site frequencies of the MSA.
            fij_target (torch.Tensor): Pairwise frequencies of the MSA.
            weights (torch.Tensor): Weights of the sequences in the MSA.
            reg_h (float, optional): Regularization parameter for the fields. Defaults to 0.0.
            reg_J (float, optional): Regularization parameter for the couplings. Defaults to 0.0.
        """
        # normalize the weights
        weights = (weights / weights.sum()).view(-1, 1, 1)
        # Compute the gradient of the energy
        prob_i = self.prob_i(X)
        grad_h = - (fi_target - (weights * prob_i).sum(0)) + 2 * reg_h * self.h
        grad_J = - (fij_target - torch.einsum("njb, nia -> iajb", weights * X, prob_i)) + 2 * reg_J * self.J
        # apply the gradinent
        self.h.grad = grad_h
        self.J.grad = grad_J                
    
            
    def fit(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 1000,
        pseudo_count: float = 0.0,
        reg_h: float = 0.0,
        reg_J: float = 0.0,
        epsconv: float = 1e-2,
    ) -> None:
        """Fits the model to the data.
        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            weights (torch.Tensor): Weights of the sequences in the MSA.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 1000.
            pseudo_count (float, optional): Pseudo-count for the frequencies. Defaults to 0.0.
            reg_h (float, optional): Regularization parameter for the fields. Defaults to 0.0.
            reg_J (float, optional): Regularization parameter for the couplings. Defaults to 0.0.
            epsconv (float, optional): Convergence threshold. Defaults to 1e-2.
        """
        # Target frequencies, if entropic order is used, the frequencies are sorted
        fi_target = get_freq_single_point(X, weights=weights, pseudo_count=pseudo_count)
        fij_target = get_freq_two_points(X, weights=weights, pseudo_count=pseudo_count)
        self.h.data = torch.log(fi_target + 1e-10)
        
        pbar = tqdm(
            total=max_epochs,
            colour="red",
            dynamic_ncols=True,
            leave=False,
            ascii="-#",
        )
        pbar.set_description(f"Loss: inf")
        prev_loss = float("inf")
        for _ in range(max_epochs):
            pbar.update(1)
            optimizer.zero_grad()
            self.compute_gradient(
                X,
                fi_target=fi_target,
                fij_target=fij_target,
                weights=weights,
                reg_J=reg_J,
                reg_h=reg_h,
            )
            optimizer.step()
            # set the autocorrelation to zero
            self.remove_autocorr()
            # compute the loss
            loss = loss_fn(
                self,
                X,
                weights=weights,
                fi_target=fi_target,
                fij_target=fij_target,
                reg_J=reg_J,
                reg_h=reg_h,
            )
            pbar.set_description(f"Loss: {loss.item():.2f}")
            if abs(prev_loss - loss.item()) < epsconv:
                break
            prev_loss = loss.item()
        pbar.close()
    
    
    def sample(
        self,
        X: torch.Tensor,
        nsweeps: int = 1000,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Samples the MSA using the model.

        Args:
            X (torch.Tensor): Input MSA one-hot encoded.
            nsweeps (int, optional): Number of sweeps to perform. Defaults to 1000.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Returns:
            torch.Tensor: Sampled MSA.
        """
        for _ in range(nsweeps):
            residue_idxs = torch.randperm(self.L)
            X = self(X, residue_idxs, beta=beta)
        
        return X
    
    
# Define the loss function
def loss_fn(
    model: nn.Module,
    X: torch.Tensor,
    weights: torch.Tensor,
    fi_target: torch.Tensor,
    fij_target: torch.Tensor,
    reg_h: float = 0.0,
    reg_J: float = 0.0,
) -> torch.Tensor:
    """Computes the negative pseudo-log-likelihood of the model.
    
    Args:
        model (nn.Module): plmDCA model.
        X (torch.Tensor): Input MSA one-hot encoded.
        weights (torch.Tensor): Weights of the sequences in the MSA.
        fi_target (torch.Tensor): Single-site frequencies of the MSA.
        fij_target (torch.Tensor): Pairwise frequencies of the MSA.
        reg_h (float, optional): Regularization parameter for the fields. Defaults to 0.0.
        reg_J (float, optional): Regularization parameter for the couplings. Defaults to 0.0.
    """
    # normalize the weights
    weights = (weights / weights.sum()).view(-1, 1)
    energy_i = - torch.einsum("ia, ia -> i", model.h, fi_target) - torch.einsum("iajb, iajb -> i", model.J, fij_target)
    logZ_i = (torch.logsumexp(model.h + torch.einsum("njb, iajb -> nia", X, model.J), dim=-1) * weights).sum(0)
    log_likelihood = - torch.sum(energy_i + logZ_i)
    
    return - log_likelihood + reg_J * torch.norm(model.J).square() + reg_h * torch.norm(model.h).square()
