import torch

class TriangularMLP(torch.nn.Module):
    def __init__(self, condition_coords, u_dim, y_dim, w=64, out_dim=None, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.y_coords = condition_coords
        self.u_coords = [i for i in range(u_dim + y_dim) if i not in self.y_coords]
        out_dim = u_dim if out_dim is None else out_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(u_dim + y_dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        #input = torch.cat((u, y), dim=1)
        #if self.time_varying:
        #    t = t.unsqueeze(-1)
        #    input = torch.cat((input, t), dim=1)
        model_out = self.net(x)

        # Embed as a triangular vector field, i.e. of the form (0, v_t)
        out = torch.zeros(x.shape[0], self.u_dim + self.y_dim, device=x.device)
        out[:, self.u_coords] = model_out

        return out