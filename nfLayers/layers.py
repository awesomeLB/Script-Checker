from torch import nn
import torch
from torch import distributions
# from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from .neural_net import NeuralNet
from torch.distributions import Transform
from torch.nn import functional as F
import numbers
import pdb

NATIVE_GRADIENT = True

class EraseCache():
    def __call__(self, layer):
        if isinstance(layer, AbstractInvertibleLayer):
            layer.erase_cache()

def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x

def sample_given_auxiliary(dist, epsilon):
    global NATIVE_GRADIENT
    if isinstance(dist, distributions.Normal):
        loc = dist.loc
        scale = dist.scale

        return loc + scale * epsilon

    if isinstance(dist, distributions.Uniform):
        low = dist.low
        high = dist.high
        if not NATIVE_GRADIENT:
            low = low.detach()
            high = high.detach()
        return low + (high - low) * epsilon
    elif isinstance(dist, distributions.Independent):
        return sample_given_auxiliary(dist.base_dist, epsilon)
    elif isinstance(dist, distributions.TransformedDistribution):
        sample = sample_given_auxiliary(dist.base_dist, epsilon)
        for t in dist.transforms:
            sample = t(sample)
        return sample
    elif isinstance(dist, MultitaskMultivariateNormal):
        raise NotImplementedError()
    elif isinstance(dist, MultivariateNormal):
        L = dist.lazy_covariance_matrix.root_decomposition().root.transpose(-2, -1)
        epsilon = epsilon.unsqueeze(-1)
        loc = dist.loc

        while loc.ndimension()<epsilon.ndimension():
            loc = loc.unsqueeze(-1)
        result = (loc + L @ epsilon).squeeze(-1)
        return result
    else:
        raise NotImplementedError(f'type {type(dist)} not supported')

class gradient_transfer_backward_transform():
    prev_val = True
    def __enter__(self):
        global NATIVE_GRADIENT
        self.prev_val = NATIVE_GRADIENT
        NATIVE_GRADIENT = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global NATIVE_GRADIENT
        NATIVE_GRADIENT = self.prev_val

def inv_softplus(x):
    if isinstance(x, numbers.Number):
        x = torch.tensor([x])
    return torch.log(torch.expm1(x))

SCALE_VALUE = 1.0
class ScaleLayer(nn.Module):

    def __init__(self, init_value=None, scale_mode='linear'):
        super().__init__()
        if init_value is None:
            init_value = SCALE_VALUE
        self.scale_mode = scale_mode
        if scale_mode == 'linear':
            self.scale = nn.Parameter(torch.tensor([init_value]))
        elif scale_mode == 'softplus':
            self.scale = nn.Parameter(inv_softplus(torch.tensor([init_value])))
        elif scale_mode == 'sigmoid':
            self.scale = nn.Parameter(torch.tensor([init_value/(1-init_value)]).log())
        else:
            raise NotImplementedError()

    def get_scale(self):
        if self.scale_mode == 'linear':
            scale = self.scale
        elif self.scale_mode == 'softplus':
            scale = F.softplus(self.scale)
        elif self.scale_mode == 'sigmoid':
            scale = torch.sigmoid(self.scale)
        return scale

    def forward(self, input):
        return self.get_scale()*input

class AbstractInvertibleLayer(Transform, nn.Module):  # , Transform):
    event_dim = 0
    grad_mode = True
    invertible=True

    def __init__(self ):
        super(AbstractInvertibleLayer, self).__init__()
        self.auxiliary_input = None
        self._cache = {'input': None, 'output': None, 'ldj': None, 'Jacob': None}

    def __hash__(self):
        return super(nn.Module, self).__hash__()

    def set_auxiliary(self, auxiliary):
        self.auxiliary_input = auxiliary

    def forward_transform(self, input, auxiliary_input=None):
        raise NotImplementedError()

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        raise NotImplementedError()

    def set_gradient_mode_transform(self, mode=True):
        # for reparameterization in PF
        self.grad_mode = mode
        for c in self.children():
            if isinstance(c, AbstractInvertibleLayer):
                c.set_gradient_mode_transform(mode)

    def forward(self, x, logp=None, auxiliary_input=None):
        raise NotImplementedError()

    def log_abs_det_jacobian(self, x, y):
        if x is self._cache['input']:
            ldj = self._cache['ldj']
        else:
            if not NATIVE_GRADIENT:
                x = self._cache['input_detach']
                x = neg_grad(x)
            y, ldj, Jacob = self.forward_transform(x, auxiliary_input=self.auxiliary_input)
        return ldj.unsqueeze(-1)

    def inv(self, y):
        if y is self._cache['output'] or not NATIVE_GRADIENT:
            assert (y-self._cache['output']).norm(-1).max()<1e-6
            if not NATIVE_GRADIENT:
                x = self._cache['input']
                Jacob = self._cache['Jacob']
                x_detach = self.clone_and_copy_data(x.detach(), y, Jacob) # for log-det jacobian
                self._cache['input_detach'] = x_detach
                x = self.clone_and_copy_data(x, y, Jacob)
            else:
                x = self._cache['input']
            return x

        x, ldj = self.backward_transform(y, auxiliary_input=self.auxiliary_input)
        self._cache.update({
            'input': x,
            'output': y,
            'ldj': ldj
        })
        self.set_gradient_mode_transform(True)
        return x

    def _inverse(self, y):
        return self.inv(y)

    def __call__(self, x):
        if x is self._cache['input']:
            return self._cache['output']
        x_input = x
        if not NATIVE_GRADIENT:
            x_input = x.detach()
        y, ldj, Jacob = self.forward_transform(x_input, auxiliary_input=self.auxiliary_input)
        if Jacob is None and not NATIVE_GRADIENT:
            print('Wagning: assessing Jacobian using autograd!')
            Jacob = torch.stack([
                torch.autograd.functional.jacobian(lambda x: self.forward_transform(x, auxiliary_input=self.auxiliary_input)[0], (x_input[i].unsqueeze(0),))[0].squeeze()
                for i in range(x_input.shape[0])],0)
        self._cache.update({
            'input': x,
            'output': y,
            'ldj': ldj,
            'Jacob': Jacob
        })
        return y

    def clone_and_copy_data(self, x, y, Jacob):
        out = jacob_mult(y, x, Jacob)
        return out

    def erase_cache(self):
        self._cache = {k: None for k in self._cache}

class NegGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    def backward(ctx, grad_output):
        return - grad_output
neg_grad = NegGrad.apply

class JacobMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, jacob):
        ctx.save_for_backward(jacob.detach())
        return x

    @staticmethod
    def backward(ctx, grad_output):
        jacob, = ctx.saved_tensors
        dtype=grad_output.dtype
        B = grad_output.unsqueeze(-1).to(torch.double)
        A = jacob.transpose(-2, -1).to(torch.double)
        sol, _ = torch.solve(B, A)
        G = sol.squeeze(-1).to(dtype)

        return G, grad_output, None
jacob_mult = JacobMult.apply

class PlanarFlow(AbstractInvertibleLayer):
    invertible = False
    def __init__(self,
                 input_size,
                 latent_size,
                 sub_network_layers,
                 sub_network_cells,
                 sub_network_activation,
                 sub_network_batch_norm,
                 biased=True,
                 scaled=True,
                 scale_mode = 'linear',
                 scale_value = None,
                 ):

        super().__init__()

        if input_size == 0:
            raise Exception('input size must be >0')
        self.input_size = input_size
        self.latent_size = latent_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.scale_value = scale_value
        self._build_layer()
        if not biased:
            raise NotImplementedError('biased must be set to true')


    def _build_layer(self):
        self.transform = NeuralNet(
            self.input_size,
            2 * self.latent_size + 1,
            n_neurons=self.sub_network_cells,
            n_layers=self.sub_network_layers,
            activation_fn=self.sub_network_activation,
            bias_last_layer=True,
            batch_norm=self.sub_network_batch_norm,
        )
        if self.scaled:
            self.scale = ScaleLayer(self.scale_value, scale_mode=self.scale_mode)
        else:
            self.scale = None

    def forward_transform(self, input: torch.Tensor, auxiliary_input=None):
        input_net = auxiliary_input

        output = self.transform(input_net)
        w = output[..., :self.latent_size]
        u = output[..., self.latent_size:2 * self.latent_size]
        b = output[..., -1:]
        uw = (u * w).sum(-1, keepdim=True)
        u = u + (F.softplus(uw) - uw - 1) * w / w.norm(2, dim=-1, keepdim=True).pow(2)
        if self.scale is not None:
            u = self.scale(u)
        r = ((input * w).sum(-1, keepdim=True) + b).tanh()
        if not self.grad_mode:
            u = u.detach()
            r = r.detach()
            self.set_gradient_mode_transform(True)
        output = input + u * r

        psi = w * (1 - r.pow(2))
        ldj = abs(1 + (u * psi).sum(-1, keepdim=False)).log()
        with torch.no_grad():
            I = torch.eye(self.latent_size, device=u.device).expand(*u.shape[:-1], self.latent_size, self.latent_size)
            Jacob = I + u.unsqueeze(-1) @ psi.unsqueeze(-2)
        return output, ldj, Jacob

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        # first order taylor approximation
        dy = output - self._cache['output']
        dx, _ = torch.solve(dy.unsqueeze(-1), self._cache['Jacob'])
        xp = self._cache['input'] + alpha*dx.squeeze(-1)
        return xp, torch.zeros_like(xp[..., 0])

class RadialFlow(AbstractInvertibleLayer):
    invertible = False
    def __init__(self,
                 input_size,
                 latent_size,
                 sub_network_layers,
                 sub_network_cells,
                 sub_network_activation,
                 sub_network_batch_norm,
                 biased=True,
                 scaled=True,
                 scale_mode = 'linear',
                 scale_value = None,
                 ):

        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.scale_value = scale_value
        self._build_layer()
        if not biased:
            raise NotImplementedError('biased must be set to true')


    def _build_layer(self):
        if self.input_size:
            self.transform = NeuralNet(
                self.input_size,
                self.latent_size + 2,
                n_neurons=self.sub_network_cells,
                n_layers=self.sub_network_layers,
                activation_fn=self.sub_network_activation,
                bias_last_layer=True,
                batch_norm=self.sub_network_batch_norm,
            )
        else:
            self.transform = None
            self.register_parameter('x0', nn.Parameter(torch.zeros(self.latent_size, requires_grad=True)))
            self.register_parameter('alpha', nn.Parameter(torch.zeros(1, requires_grad=True)))
            self.register_parameter('beta', nn.Parameter(torch.zeros(1, requires_grad=True)))

        self.register_buffer('bias_softplus', inv_softplus(1.0))
        if self.scaled:
            self.scale = ScaleLayer(self.scale_value)
        else:
            self.scale = None

    def forward_transform(self, input: torch.Tensor, auxiliary_input=None):
        if self.transform is not None:
            input_net = auxiliary_input
            embed = self.transform(input_net)
            if self.scale is not None:
                embed = self.scale(embed)
            x0 = embed[..., :self.latent_size]

            alpha = F.softplus(embed[..., -2]+self.bias_softplus).unsqueeze(-1)
            beta = -alpha + F.softplus(embed[..., -1]+self.bias_softplus).unsqueeze(-1)
        else:
            x0 = self.x0
            alpha = F.softplus(self.alpha+self.bias_softplus).unsqueeze(-1)
            beta = -alpha + F.softplus(self.beta+self.bias_softplus).unsqueeze(-1)
        r = (input - x0).norm(dim=-1, keepdim=True)
        h = 1 / (alpha + r)
        output = input + beta * h * (input - x0)

        n = input.shape[-1]
        ldj = (n-1) * torch.log1p(beta*h) + torch.log1p(beta*h - beta*r*h.pow(2))
        with torch.no_grad():
            I = torch.eye(input.shape[-1], device=input.device).expand(*input.shape, input.shape[-1])
            Jacob = I * (1+beta * h).unsqueeze(-1) - \
                               (beta * h.pow(2)).unsqueeze(-1) * (input-x0).unsqueeze(-1) @ (input-x0).unsqueeze(-2) / r.unsqueeze(-1)
        return output, ldj, Jacob

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        dy = output - self._cache['output']
        dx, _ = torch.solve(dy.unsqueeze(-1), self._cache['Jacob'])
        xp = self._cache['input'] + alpha*dx.squeeze(-1)
        return xp, torch.zeros_like(xp[..., 0])

class AffineFlow(AbstractInvertibleLayer):
    invertible = True
    def __init__(self,
                 input_size,
                 latent_size,
                 sub_network_layers,
                 sub_network_cells,
                 sub_network_activation,
                 sub_network_batch_norm,
                 biased=True,
                 scaled=True,
                 scale_mode = 'linear',
                 scale_value = None,
                 ):

        super().__init__()

        if input_size == 0:
            raise Exception('input size must be >0')
        self.input_size = input_size
        self.latent_size = latent_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.scale_value = scale_value
        self._build_layer()
        if not biased:
            raise NotImplementedError('biased must be set to true')


    def _build_layer(self):
        self.transform = NeuralNet(
            self.input_size,
            2 * self.latent_size,
            n_neurons=self.sub_network_cells,
            n_layers=self.sub_network_layers,
            activation_fn=self.sub_network_activation,
            bias_last_layer=True,
            batch_norm=self.sub_network_batch_norm,
        )
        self.register_buffer('bias_softplus', inv_softplus(1.0))
        if self.scaled:
            self.scale = ScaleLayer(self.scale_value, scale_mode=self.scale_mode)
        else:
            self.scale = None

    def forward_transform(self, input: torch.Tensor, auxiliary_input=None):
        input_net = auxiliary_input

        tsf = self.transform(input_net)
        if self.scaled:
            tsf = self.scale(tsf)
        loc = tsf[..., :self.latent_size]
        scale = F.softplus(self.bias_softplus+tsf[..., self.latent_size:])

        ldj = scale.log().sum(-1, keepdim=False).expand(input.shape[:-1])
        output = input*scale + loc
        return output, ldj, scale.diag_embed()

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        input_net = auxiliary_input

        tsf = self.transform(input_net)
        if self.scaled:
            tsf = self.scale(tsf)
        loc = tsf[..., :self.latent_size]
        scale = F.softplus(self.bias_softplus+tsf[..., self.latent_size:])

        ldj = scale.log().sum(-1, keepdim=False).expand(output.shape[:-1])
        input = (output-loc)/scale.clamp_min(1e-8)
        return input, ldj

class MirroredCouplingFlow(AbstractInvertibleLayer):
    invertible = True
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.c1 = CouplingFlow(*args, **kwargs)
        partition_size = self.c1.dim_split_b
        self.c2 = CouplingFlow(*args, partition_size=partition_size, **kwargs)

        self.c2.idx_b.data.copy_(self.c1.idx_a.data)
        self.c2.idx_a.data.copy_(self.c1.idx_b.data)
        rp = torch.cat([self.c2.idx_a, self.c2.idx_b], 0)
        rp_inv = rp.argsort()
        self.c2.inverse_idx.data.copy_(rp_inv)

    def forward_transform(self, x, auxiliary_input=None):
        x_bis = self.c1(x)
        ldj1 = self.c1.log_abs_det_jacobian(x, x_bis)
        y = self.c2(x_bis)
        ldj2 = self.c2.log_abs_det_jacobian(x_bis, y)
        return y, ldj1+ldj2, None

    def backward_transform(self, y, auxiliary_input=None):
        x_bis = self.c2.inv(y)
        ldj2 = self.c2.log_abs_det_jacobian(x_bis, y)
        x = self.c1.inv(x_bis)
        ldj1 = self.c1.log_abs_det_jacobian(x, x_bis)
        return x, ldj1+ldj2

class EmptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0

class AffineTransformNF(AbstractInvertibleLayer):
    invertible = True
    def __init__(self, loc, scale):
        super().__init__()
        self.register_buffer('loc', loc)
        self.register_buffer('scale', scale)

    def forward_transform(self, input, auxiliary_input=None):
        scale = self.scale
        ldj = scale.log().sum(-1, keepdim=False).expand(input.shape[:-1])
        output = input*scale + self.loc
        return output, ldj, scale.diag_embed()

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        scale = self.scale
        ldj = scale.log().sum(-1, keepdim=False).expand(output.shape[:-1])
        input = (output-self.loc)/scale.clamp_min(1e-8)
        return input, ldj



class ActNorm(AbstractInvertibleLayer):
    invertible = True
    def __init__(self, input_size):
        super().__init__()
        self.register_parameter('loc', nn.Parameter(torch.zeros(input_size)))
        self.register_parameter('scale', nn.Parameter(torch.zeros(input_size)+inv_softplus(1.0)))
        self.initialized=False
        print('actnorm instantiated')

    def forward_transform(self, input, auxiliary_input=None):
        if not self.initialized:
            with torch.no_grad():
                sd = input.reshape(-1, input.shape[-1]).std(dim=0, keepdim=False)
                s = inv_softplus(1/sd)
                m = (-(input.reshape(-1, input.shape[-1]) / sd).mean(dim=0, keepdim=False))
            self.loc.data.copy_(m)
            self.scale.data.copy_(s)
        self.initialized = True

        scale = F.softplus(self.scale)
        ldj = scale.log().sum(-1, keepdim=False).expand(input.shape[:-1])
        output = input*scale + self.loc
        return output, ldj, scale.diag_embed()

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        if not self.initialized:
            with torch.no_grad():
                m = output.reshape(-1, output.shape[-1]).mean(0)
                s = inv_softplus(output.reshape(-1, output.shape[-1]).std(0))
                self.initialized=True
            self.loc.data.copy_(m)
            self.scale.data.copy_(s)
        scale = F.softplus(self.scale)
        ldj = scale.log().sum(-1, keepdim=False).expand(output.shape[:-1])
        input = (output-self.loc)/(scale+1e-8)
        return input, ldj

class HouseHolderFlow(AbstractInvertibleLayer):
    invertible = True
    scaling_householder = False
    def __init__(self,
                 input_size,
                 latent_size,
                 sub_network_layers,
                 sub_network_cells,
                 sub_network_activation,
                 sub_network_batch_norm,
                 biased=True,
                 scaled=True,
                 scale_mode = 'linear',
                 scale_value = None,
                 ):

        super().__init__()

        if input_size == 0:
            raise Exception('input size must be > 0')
        self.input_size = input_size
        self.latent_size = latent_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.scale_value = scale_value
        self._build_layer()
        if not biased:
            raise NotImplementedError('biased must be set to true')


    def _build_layer(self):
        self.transform = NeuralNet(
            self.input_size,
            self.latent_size if not self.scaling_householder else 2*self.latent_size,
            n_neurons=self.sub_network_cells,
            n_layers=self.sub_network_layers,
            activation_fn=self.sub_network_activation,
            bias_last_layer=True,
            batch_norm=self.sub_network_batch_norm,
        )
        self.register_buffer('bias_softplus', inv_softplus(1.0))

        if self.scaled:
            self.scale = ScaleLayer(self.scale_value)
        else:
            self.scale = None

    def forward_transform(self, input: torch.Tensor, auxiliary_input=None):
        input_net = auxiliary_input

        v = self.transform(input_net)
        if self.scaling_householder:
            s = v[..., self.latent_size:]
            if self.scale:
                s = self.scale(s)
            s = F.softplus(self.bias_softplus+s)
            v = v[..., :self.latent_size]
        v = v.unsqueeze(-1)
        with torch.no_grad():
            H = 2 * v @ v.transpose(-2, -1) / v.norm(dim=-2, keepdim=True).pow(2)
            H = torch.eye(v.shape[-2], device=v.device).expand_as(H) - H
        vz = (v.transpose(-2, -1) @ input.unsqueeze(-1))
        output = input - (2 * v @ vz).squeeze(-1)/v.norm(dim=-2, keepdim=False).pow(2)

        if self.scaling_householder:
            output = s*output
            ldj = s.log().sum(-1).view_as(input[..., 0])
        else:
            ldj = torch.zeros_like(input[..., 0])
        return output, ldj, H.detach()

    def backward_transform(self, output, auxiliary_input=None, alpha=1.0):
        input_net = auxiliary_input

        v = self.transform(input_net)
        if self.scaling_householder:
            s = v[..., self.latent_size:]
            if self.scale:
                s = self.scale(s)
            s = F.softplus(self.bias_softplus+s)
            v = v[..., :self.latent_size]
        v = v.unsqueeze(-1)
        H = 2 * v @ v.transpose(-2, -1) / v.norm(dim=-2, keepdim=True).pow(2)
        H = torch.eye(v.shape[-2], device=v.device).expand_as(H) - H
        if self.scaling_householder:
            output = output/s
        input = torch.solve(output.unsqueeze(-1), H)[0].squeeze(-1)

        if self.scaling_householder:
            ldj = s.log().sum(-1).view_as(input[..., 0])
        else:
            ldj = torch.zeros_like(input[..., 0])
        return input, ldj

class ScaledHouseHolderFlow(HouseHolderFlow):
    scaling_householder=True

class CouplingFlow(AbstractInvertibleLayer):
    invertible = True
    def __init__(self,
                 input_size,
                 output_size,
                 sub_network_layers,
                 sub_network_cells,
                 sub_network_activation,
                 sub_network_batch_norm,
                 scaled=True,
                 partition_size=None,
                 biased=True,
                 scale_mode = 'linear',
                 scale_value = None,
                 idx_a = None,
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sub_network_layers = sub_network_layers
        self.sub_network_cells = sub_network_cells
        self.sub_network_activation = sub_network_activation
        self.sub_network_batch_norm = sub_network_batch_norm
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.scale_value = scale_value
        self.biased = biased

        if partition_size is None:
            partition_size = self.output_size // 2
        self.dim_split_a = partition_size
        self.input_size = input_size + partition_size

        self.dim_split_b = self.output_size - partition_size
        if idx_a is None:
            rp = torch.randperm(self.output_size)
            idx_a = rp[:self.dim_split_a]
            idx_b = rp[self.dim_split_a:]
            inverse_idx = rp.argsort()
        else:
            idx_b = torch.tensor([i for i in range(output_size) if i not in idx_a])
            idx_a = torch.tensor(idx_a)
            inverse_idx = torch.cat([idx_a, idx_b]).argsort()
        self.register_buffer('idx_a', idx_a)
        self.register_buffer('idx_b', idx_b)
        self.register_buffer('inverse_idx', inverse_idx)

        self._build_layer()

    def _build_layer(self):
        self.transform_ft = NeuralNet(
            self.input_size,
            self.dim_split_b * 2 if self.biased else self.dim_split_b,
            n_neurons=self.sub_network_cells,
            n_layers=self.sub_network_layers,
            activation_fn=self.sub_network_activation,
            bias_last_layer=True,
            batch_norm=self.sub_network_batch_norm,
            last_layer=ScaleLayer(self.scale_value, self.scale_mode) if self.scaled else None
        )
        self.register_buffer('bias_softplus', inv_softplus(1.0))

    def forward_transform(self, input, auxiliary_input=None):
        input_a, input_b = input[..., self.idx_a], input[..., self.idx_b]
        if auxiliary_input is not None:
            ndimdiff = input_a.ndimension() - auxiliary_input.ndimension()
            if ndimdiff>0:
                auxiliary_input = auxiliary_input.expand(*input_a.shape[:-1], auxiliary_input.shape[-1])
            input_net = torch.cat([
                input_a,
                auxiliary_input
            ], dim=-1)
        else:
            input_net = input_a

        f = self.transform_ft(input_net)
        if self.biased:
            f, t = f[..., :self.dim_split_b], f[..., self.dim_split_b:]
        else:
            t = 0
        f = F.softplus(f+self.bias_softplus)

        if not self.grad_mode:
            f = f.detach()
            if t is not 0:
                t = t.detach()
            self.set_gradient_mode_transform(True)

        output = torch.cat([
            input_a,
            input_b * f + t
        ], dim=-1)

        Jacob = None

        output = output[..., self.inverse_idx]
        ldj = f.log().sum(dim=-1, keepdim=False)
        return output, ldj, Jacob

    def backward_transform(self, output,
                           auxiliary_input=None, alpha=1.0):
        input_a = output[..., self.idx_a]
        if auxiliary_input is not None:
            ndimdiff = input_a.ndimension() - auxiliary_input.ndimension()
            if ndimdiff>0:
                auxiliary_input = auxiliary_input.expand(*input_a.shape[:-1], auxiliary_input.shape[-1])
            input_net = torch.cat([
                input_a,
                auxiliary_input
            ], dim=-1)
        else:
            input_net = input_a
        f = self.transform_ft(input_net)
        if self.biased:
            f, t = f[..., :self.dim_split_b], f[..., self.dim_split_b:]
        else:
            t = 0
        f = F.softplus(f+self.bias_softplus)
        output_b = output[..., self.idx_b]
        if not self.grad_mode:
            f = f.detach()
            if t is not 0:
                t = t.detach()
            self.set_gradient_mode_transform(True)
        input_b = (output_b - t) / f
        input = torch.cat([
            input_a,
            input_b
        ], dim=-1)
        input = input[..., self.inverse_idx]

        ldj = f.log().sum(dim=-1, keepdim=False)
        return input, ldj





