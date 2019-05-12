
import torch
import torch.nn as nn
import numpy as np
import itertools
from scipy.integrate import ode
import sys
import matplotlib.pylab as plt

Nx_trg = 2 # the dimension of target sysstem

class XGenerator(nn.Module):
    def __init__(self, Nz, Nx, Nlayer, Nh):
        super(XGenerator, self).__init__()
        
        arg = (nn.Linear(Nz, Nh), nn.Tanh())
        for k1 in range(Nlayer):
            arg += (nn.Linear(Nh, Nh), nn.Tanh())
        arg += (nn.Linear(Nh, Nx),)
        self.net_generator = nn.Sequential(*arg)
    def forward(self, _Z):
        # _Z: (*, Nz)
        _X = self.net_generator(_Z)
        return _X

def test001():
    Nz = 10
    Nx = 2
    Nlayer = 2
    Nh = 16
    Nbatch = 32

    x_generator = XGenerator(Nz, Nx, Nlayer, Nh)
    _Z = torch.randn(Nbatch, Nz)
    _X = x_generator(_Z)
    assert _X.shape == (Nbatch, Nx)


def z_generator(Nbatch, Nz):
    while True:
        yield (np.random.rand(Nbatch, Nz).astype(np.float32) - 1/2)*2 * np.sqrt(3)

def test002():
    Nbatch = 32
    Nz = 10
    g = z_generator(Nbatch, Nz)
    for Z in itertools.islice(g, 10):
        assert Z.shape == (Nbatch, Nz)
        assert isinstance(Z[0,0], np.float32)

#http://www.scholarpedia.org/article/Duffing_oscillator
def DuffingEq(t, y):
    # y[0]: position, y[1]: velocity
    alpha, beta, delta, gamma, omg = 1., 0., 0.05, 7.5, 1.0    
    return [
        y[1],
        -delta * y[1] - beta * y[0] - alpha * (y[0]**3) + gamma * np.cos(omg * t),
    ]

def test003():
    r = ode(DuffingEq)
    x0 = np.array([0,0])
    r.set_initial_value(x0,0)
    dt = np.pi * 2
    X = np.stack([r.integrate(t+dt) for t in np.arange(0., 2*np.pi * (2**10), dt)], axis=0)
    plt.figure()
    plt.plot(X[:,0], X[:,1], '.')
    plt.show()

    print(np.mean(X, axis=0), np.std(X, axis=0))

def operate_poincare_map(X0, T = 2*np.pi):
    # X0: (*, Nx_trg)
    # return X1: (*, Nx_trg)

    mu = [2.59164693, 0.30891403]
    sd = [0.46986201, 2.15622998]

    X0_raw = X0 * sd + mu 

    X1_raw = []
    r = ode(DuffingEq)
    for k1 in range(X0_raw.shape[0]):
        r.set_initial_value(X0_raw[k1,:], 0)
        x1_raw = r.integrate(T)
        X1_raw.append(x1_raw)
    X1_raw = np.stack(X1_raw, axis=0)
    X1 = (X1_raw - mu)/sd
    X1 = X1.astype(np.float32)

    return X1

def test004():
    Nbatch = 2**10
    X0 = np.random.randn(Nbatch, Nx_trg) + [1, 0]
    X1 = operate_poincare_map(X0)
    assert X1.shape == (Nbatch, Nx_trg)
    assert isinstance(X1[0,0], np.float32)
    plt.figure()
    plt.plot(X0[:,0], X0[:,1], '.', label='X0')
    plt.plot(X1[:,0], X1[:,1], '.', label='X1')
    plt.legend()
    plt.show()

def robust_sinkhorn_iteration(_M, _p, _q, _eps, tol, max_itr = 2**12):
    _alpha = _p * 0
    _beta = _q * 0
    cnt = 0
    
    assert max(_M.shape) <= 2**6, "The shape of M is %s. That exceeds the limitation: 64" % str(_M.shape)

    while True:
        
        _P = torch.exp(-(_M-_alpha-_beta)/_eps -1)
        _qhat = torch.sum(_P, dim=0, keepdim=True)
        _err = torch.sum(torch.abs(_qhat - _q))

        if _err < tol or cnt >= max_itr:
            break
        else:
            cnt += 1

        _delta_row = torch.min(_M - _alpha, dim=0, keepdim = True)[0]
        _beta = _eps + _eps * torch.log(_q) + _delta_row             - _eps * torch.log( torch.sum( torch.exp(-(_M-_alpha-_delta_row)/_eps ), dim=0, keepdim = True ) )
        _delta_col = torch.min(_M - _beta, dim=1, keepdim = True)[0]
        _alpha = _eps + _eps * torch.log(_p) + _delta_col             - _eps * torch.log( torch.sum( torch.exp( -(_M - _beta - _delta_col)/_eps  ), dim=1, keepdim = True )  )

    if cnt == max_itr:
        #print('Warning: Sinkhorn iteration did not converge within the given max iteration number: %d.' % max_itr)
        pass
    _dist = torch.sum(_p * _alpha) + torch.sum(_q * _beta) - _eps
    return _dist, cnt

def measure_distance(_X0, _X1, tol = 1e-4, eps_given = 1e-2, max_itr = 2**10):
    # _X0: (*, Nx), _X1: (*, Nx)
        
    _M01 = torch.sum(torch.abs(_X0.unsqueeze(1) - _X1), dim=2)
    _M00 = torch.sum(torch.abs(_X0.unsqueeze(1) - _X0), dim=2)
    _M11 = torch.sum(torch.abs(_X1.unsqueeze(1) - _X1), dim=2)

    _p = 1/_X0.shape[0] * torch.ones(_X0.shape[0])
    _q = 1/_X1.shape[0] * torch.ones(_X1.shape[0])

    _eps = torch.tensor(eps_given)
        
    _dist00, cnt_00 = robust_sinkhorn_iteration(_M00, _p.unsqueeze(1), _p.unsqueeze(0), _eps, tol, max_itr=max_itr)
    _dist11, cnt_11 = robust_sinkhorn_iteration(_M11, _q.unsqueeze(1), _q.unsqueeze(0), _eps, tol, max_itr=max_itr)
    _dist01, cnt_01 = robust_sinkhorn_iteration(_M01, _p.unsqueeze(1), _q.unsqueeze(0), _eps, tol, max_itr=max_itr)
    
    _wdist = 2. * _dist01 - _dist00 - _dist11
    
    return _wdist, (cnt_00, cnt_11, cnt_01)


def test005():
    _X0 = torch.randn(2**5, 3)
    _X1 = torch.randn(2**5+1, 3) 

    measure_distance(_X0, _X1)

def call_training(x_generator, z_generator, Nitr, optimizer = None, eps_given = 1e-2, max_itr=2**10):
    # z_generator: generator
    # x_generator: instance of XGenerator
    if optimizer is None:
        optimizer = torch.optim.Adam(x_generator.parameters())
    training_hist = []
    
    for k1, Z in enumerate(itertools.islice(z_generator, Nitr)):
        _Z = torch.from_numpy(Z) # (*, Nz)
        _X0 = x_generator(_Z) # (*, Nx)
        X0 = _X0.data.numpy() 
        X1 = operate_poincare_map(X0)
        _X1 = torch.from_numpy(X1) # (*, Nx)
        _wdist, (cnt_00, cnt_11, cnt_01) = measure_distance(_X0, _X1, 
            eps_given = eps_given, max_itr = max_itr)
        
        training_hist.append(float(_wdist))
        sys.stdout.write('itr %04d wdist %8.2e s-iteration: (%4d, %4d, %4d)\r' % (k1, float(_wdist), cnt_00, cnt_11, cnt_01))
        
        x_generator.zero_grad()
        _wdist.backward()
        optimizer.step()
    return training_hist

def test006():

    Nz, Nlayer, Nh = 2, 2, 10
    x_generator = XGenerator(Nz, Nx_trg, Nlayer, Nh)

    Nbatch = 2**5
    z_generator_ = z_generator(Nbatch, Nz)

    Nitr = 2**3
    call_training(x_generator, z_generator_, Nitr)

if __name__ == "__main__":
    test001()
    test002()
    test003()
    test004()
    test005()
    test006()
