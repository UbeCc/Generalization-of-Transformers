import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

def build_model(conf):
    if conf.family == "transformer":
        model = Transformer(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "kan":
        assert conf.n_dims == 1, "Only supporting dims=1 for KAN"
        model = KAN(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
        )
    elif conf.family == 'mlp':
        assert conf.n_dims == 1, "Only supporting dims=1 for MLP"
        model = MLP(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
        )
    else:
        raise NotImplementedError

    return model

class MLP(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128):
        super(MLP, self).__init__()
        self.n_dims = n_dims
        self.n_positions = n_positions
        input_dim = n_positions * 2 - 1
        self.name = f"mlp_in={input_dim}_hidden={n_embd}"
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_positions)
        )

    def forward(self, xs, ys, inds=None, use_pad=False):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")                
        bsize, points, dim = xs.shape
        assert dim == 1, "Only support dims=1"
        xs = xs[:, :, 0]
        input = torch.empty(torch.Size((bsize, 2 * points - 1)))
        xy_interleaved = torch.stack([xs[:, :-1], ys[:, :-1]], dim=-1).reshape(bsize, -1)
        input[:, :-1] = xy_interleaved
        input[:, -1] = xs[:, -1]
        # input = torch.stack([xs, ys], dim=-1).reshape(bsize, -1)
        return self.mlp(input)[:, :, 0][:, inds]
    
# ref: https://github.com/dtsip/in-context-learning.git
class Transformer(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(Transformer, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs, ys, pad=None):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs.shape
        if pad:
            pad_wide = torch.cat(
                (
                    pad.view(bsize, points, 1),
                    torch.zeros(bsize, points, dim - 1, device=pad.device),
                ),
                axis=2,
            )
            
        ys_wide = torch.cat(
            (
                ys.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys.device),
            ),
            axis=2,
        )
        if pad:
            zs = torch.stack((xs, pad_wide, ys_wide), dim=2)
            zs = zs.view(bsize, 3 * points, dim)
        else:
            zs = torch.stack((xs, ys_wide), dim=2)
            zs = zs.view(bsize, 2 * points, dim)
            
        return zs

    def forward(self, xs, ys, inds=None, use_pad=False):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        if use_pad:            
            pad = torch.zeros(ys.shape).to(ys.device)
            zs = self._combine(xs, ys, pad)
            embeds = self._read_in(zs)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            prediction = self._read_out(output)
            return prediction[:, ::2, 0][:, inds]  # predict only on xs
        else:
            zs = self._combine(xs, ys)
            embeds = self._read_in(zs)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            prediction = self._read_out(output)
            return prediction[:, ::2, 0][:, inds] # bsize, 2 * points, 1
        
# ref: https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features, f'{x.shape}, {self.in_features}'
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128):
        super(KAN, self).__init__()
        self.n_positions = n_positions
        self.n_dims = n_dims
        input_dim = n_positions * 2 - 1
        layers_hidden = [input_dim, n_embd, n_embd, n_positions]
        self.grid_size = 5
        self.spline_order = 3
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=5,
                    spline_order=3,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=nn.ReLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1],
                )
            )

    def forward(self, xs, ys, inds=None, update_grid=True):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")                
        bsize, points, dim = xs.shape
        assert dim == 1, "Only support dims=1"
        xs = xs[:, :, 0]
        input = torch.empty(torch.Size((bsize, 2 * points - 1)))
        xy_interleaved = torch.stack([xs[:, :-1], ys[:, :-1]], dim=-1).reshape(bsize, -1)
        input[:, :-1] = xy_interleaved
        input[:, -1] = xs[:, -1]
        for layer in self.layers:
            if update_grid:
                layer.update_grid(input)
            input = layer(input)
        return input

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )