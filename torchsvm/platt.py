import numpy as np
import torch
import matplotlib.pyplot as plt

class PlattScalerTorch:
    """
    Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
    Fits A,B with regularized logistic regression on decision values f and labels y∈{-1,1}.
    Uses Newton updates with damping and target smoothing per Platt (1999).
    """
    def __init__(self, max_iter=100, tol=1e-8, reg=1e-6, dtype=torch.double, device=None):
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.A = None
        self.B = None

    @torch.no_grad()
    def fit(self, f, y):
        """
        f: (n,) decision values (raw scores), torch tensor
        y: (n,) labels in {-1,1}, torch tensor
        """
        f = f.reshape(-1).to(self.device, self.dtype)
        y = y.reshape(-1).to(self.device, self.dtype)

        # convert to targets t in {0,1} and apply target smoothing from Platt
        # t+ = (N+ + 1) / (N+ + 2), t- = 1 / (N- + 2)
        pos = (y > 0)
        npos = pos.sum().item()
        nneg = y.numel() - npos
        t_pos = (npos + 1.0) / (npos + 2.0)
        t_neg = 1.0 / (nneg + 2.0)
        t = torch.where(pos, torch.tensor(t_pos, dtype=self.dtype, device=self.device),
                              torch.tensor(t_neg, dtype=self.dtype, device=self.device))

        # initialize A,B
        A = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        ratio = torch.tensor((nneg + 1e-12) / (npos + 1e-12),
                     dtype=self.dtype, device=self.device)
        B = torch.log(ratio)
        # B = torch.tensor(torch.log((nneg + 1e-12)/(npos + 1e-12)), dtype=self.dtype, device=self.device)

        # Newton updates
        for _ in range(self.max_iter):
            # p = sigmoid(A*f + B)
            z = A * f + B
            # numerically stable sigmoid
            p = torch.where(z >= 0, 1.0 / (1.0 + torch.exp(-z)), torch.exp(z) / (1.0 + torch.exp(z)))

            # logloss with small L2 on A,B (reg)
            # gradient wrt A,B
            w = p * (1.0 - p)                 # Hessian weights
            # g = X^T (p - t) + reg*theta
            gA = torch.sum((p - t) * f) + self.reg * A
            gB = torch.sum(p - t) + self.reg * B

            # Hessian (2x2)
            HAA = torch.sum(w * f * f) + self.reg
            HAB = torch.sum(w * f)              # == HBA
            HBB = torch.sum(w) + self.reg

            # Solve for step: H * [dA, dB]^T = [gA, gB]^T
            det = HAA * HBB - HAB * HAB
            if det.abs() < 1e-24:
                # fallback small step if nearly singular
                stepA = -gA / (HAA + 1e-12)
                stepB = -gB / (HBB + 1e-12)
            else:
                stepA = -( HBB * gA - HAB * gB) / det
                stepB = -(-HAB * gA + HAA * gB) / det

            # damped update to ensure progress
            damping = 1.0
            for _inner in range(10):
                A_new = A + damping * stepA
                B_new = B + damping * stepB

                # check improvement via approximate line-search on NLL
                z_new = A_new * f + B_new
                p_new = torch.where(z_new >= 0, 1.0 / (1.0 + torch.exp(-z_new)),
                                              torch.exp(z_new) / (1.0 + torch.exp(z_new)))
                # NLL with reg
                eps = 1e-12
                nll_new = -torch.sum(t * torch.log(p_new + eps) + (1.0 - t) * torch.log(1.0 - p_new + eps))
                nll_new = nll_new + 0.5 * self.reg * (A_new*A_new + B_new*B_new)

                z_old = z
                p_old = p
                nll_old = -torch.sum(t * torch.log(p_old + eps) + (1.0 - t) * torch.log(1.0 - p_old + eps))
                nll_old = nll_old + 0.5 * self.reg * (A*A + B*B)

                if nll_new <= nll_old + 1e-12:
                    A, B = A_new, B_new
                    break
                damping *= 0.5

            # convergence on parameter step
            if (torch.abs(stepA) + torch.abs(stepB)) < self.tol:
                break

        self.A = A
        self.B = B
        return self

    @torch.no_grad()
    def predict_proba(self, f):
        f = torch.as_tensor(f, dtype=self.dtype, device=self.device).reshape(-1)
        z = self.A * f + self.B
        p1 = torch.where(z >= 0, 1.0 / (1.0 + torch.exp(-z)), torch.exp(z) / (1.0 + torch.exp(z)))
        # Return [P(y=-1), P(y=1)] per row to match sklearn style
        return torch.stack([1.0 - p1, p1], dim=1)
    
    @torch.no_grad()
    def reliability_curve(self, y_true, p_pred, n_bins=15):
        """
        y_true: tensor/array in {-1,1}
        p_pred: predicted prob P(y=1|x) in [0,1]
        returns: bin_centers, mean_pred, frac_pos, counts
        """
        y = torch.as_tensor(y_true).reshape(-1)
        if y.dtype != torch.float64 and y.dtype != torch.float32:
            y = y.double()
        y01 = (y > 0).double()

        p = torch.as_tensor(p_pred).reshape(-1).double()
        p = torch.clamp(p, 1e-8, 1-1e-8)

        edges = torch.linspace(0., 1., steps=n_bins+1)
        idx = torch.bucketize(p, edges, right=True) - 1
        idx = torch.clamp(idx, 0, n_bins-1)

        mean_pred = torch.zeros(n_bins, dtype=torch.double)
        frac_pos  = torch.zeros(n_bins, dtype=torch.double)
        counts    = torch.zeros(n_bins, dtype=torch.long)

        for b in range(n_bins):
            mask = (idx == b)
            cnt = mask.sum()
            counts[b] = cnt
            if cnt > 0:
                mean_pred[b] = p[mask].mean()
                frac_pos[b]  = y01[mask].mean()

        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        return bin_centers.numpy(), mean_pred.numpy(), frac_pos.numpy(), counts.numpy()

    def expected_calibration_error(self, mean_pred, frac_pos, counts):
        n = counts.sum()
        w = counts / max(n, 1)
        return float(np.sum(w * np.abs(frac_pos - mean_pred)))

    def brier_score(self, y_true, p_pred):
        y01 = (np.array(y_true).reshape(-1) > 0).astype(float)
        p = np.clip(np.array(p_pred).reshape(-1), 1e-8, 1-1e-8)
        return float(np.mean((p - y01)**2))

    def plot_calibration(self, bin_centers, mean_pred, frac_pos, counts, label="Platt", show_counts=True):
        plt.figure(figsize=(5.2, 5.2), dpi=140)
        plt.plot([0,1], [0,1], linestyle="--", linewidth=1.5, label="Perfect")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2.0, label=label)
        plt.xlabel("Predicted probability (bin average)", fontsize=12)
        plt.ylabel("Observed frequency (empirical)", fontsize=12)
        plt.title("Calibration (Reliability) Curve", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.legend()
        if show_counts:
            # annotate per-bin counts
            for x, y, c in zip(mean_pred, frac_pos, counts):
                if c > 0:
                    plt.text(x, y, f"{int(c)}", fontsize=8, ha="center", va="bottom")
        plt.tight_layout()
        plt.show()

