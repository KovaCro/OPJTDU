import torch

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        grid_range=[-1, 1],
        spline_order=3,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

    def splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for p in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(p + 1)])/(grid[:, p:-1] - grid[:, : -(p + 1)])*bases[:, :, :-1]) + ((grid[:, p + 1 :] - x)/(grid[:, p + 1 :] - grid[:, 1:(-p)])* bases[:, :, 1:])
        return bases.contiguous()

    def forward(self, x):
        tmp = x.shape
        x = x.reshape(-1, self.in_features)
        base_activation = torch.nn.SiLU()
        base_output = torch.nn.functional.linear(base_activation(x), self.base_weight)
        spline_output = torch.nn.functional.linear(
            self.splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*tmp[:-1], self.out_features)
        return output


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers,
        grid_size=5,
        grid_range=[-1, 1],
        spline_order=3,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers, layers[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x