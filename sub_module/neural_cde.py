"""
Implementation of Neural CDE

References:
    NeuralCDE: https://github.com/patrick-kidger/NeuralCDE
"""
import torch
import torchcde


class CustomCubicSpline(torchcde.interpolation_cubic.CubicSpline):
    """
    CustomCubicSpline overrides derivative function

    Methods:
        derivative: calculate derivative for α and β

    """

    def derivative(self, t):
        """
        Our derivative function returns the tuple of derivative for α and β

        Args:
            t: one dimensional tensor of times

        Returns:
            deriv: derivative for α
            deriv: derivative for β

        """
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part

        return deriv, deriv


class NeuralCDE(torch.nn.Module):

    def __init__(self, cde_func, input_size, hidden_size, output_size, n_factors, ode_method='rk4'):
        super(NeuralCDE, self).__init__()
        self.cde_func = cde_func
        self.output_size = output_size
        self.n_factors = n_factors
        self.ode_method = ode_method

        self.init_alpha0 = torch.nn.Linear(input_size, hidden_size)
        self.init_beta0 = torch.nn.Linear(input_size, hidden_size)
        self.init_factors0 = torch.nn.Linear(input_size, hidden_size)

        self.readout_alpha = torch.nn.Linear(hidden_size, output_size)
        self.readout_beta = torch.nn.Linear(hidden_size, output_size * n_factors)

    def forward(self, coeffs):

        X = CustomCubicSpline(coeffs)

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        # z0 → alpha_0, beta_0
        X0 = X.evaluate(X.interval[0])
        alpha_0 = self.init_alpha0(X0)
        beta_0 = self.init_beta0(X0)

        ######################+
        # Actually solve the CDE.
        ######################
        # The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        # This will be a tensor of shape (..., len(t), hidden_channels).
        z_alpha_T, z_beta_T = torchcde.cdeint(X=X, func=self.cde_func, z0=(alpha_0, beta_0), t=X.grid_points,
                                              method=self.ode_method)

        # [batch, timestep, hidden] -> [timestep, batch, hidden]
        z_alpha_T = z_alpha_T.transpose(0, 1)
        z_beta_T = z_beta_T.transpose(0, 1)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_alpha_T = z_alpha_T[-1]
        z_beta_T = z_beta_T[-1]

        alpha_T = self.readout_alpha(z_alpha_T)
        beta_T = self.readout_beta(z_beta_T)

        return alpha_T, beta_T
