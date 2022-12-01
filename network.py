import torch, math
from collections import OrderedDict
import threading


class Tanh(torch.nn.Module):
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, d=False):
        if d:
            return (x > 0) * (1 - torch.tanh(self.alpha * x)**2) * self.alpha
        return (x > 0) * torch.tanh(self.alpha * x)
    

class Softsign(torch.nn.Module):
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, d=False):
        y = self.alpha * x
        if d:
            return (x > 0) * self.alpha / (torch.abs(y) + 1) ** 2
        return (x > 0) * y / (torch.abs(y) + 1)
    
    
class Softsign2(torch.nn.Module):
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        y = (self.alpha * x) ** 2
        return (x > 0) * y / (y + 1)


class History:

    def __init__(self):
        self.items = []
        
    def __call__(self):
        return torch.stack(self.items)

    def append(self, item):
        self.items.append(item)

    def reset(self):
        self.items = []


class Sensory(torch.nn.Module):

    def __init__(self, n_neurons, transmitter):
        super().__init__()
        self.n_neurons = n_neurons
        self.transmitter = torch.nn.Parameter(torch.tensor(transmitter, dtype=torch.float32), requires_grad=False)
        self.r = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False)
        
        self.history = History()
        
    def rt(self):
        return torch.outer(self.r, self.transmitter)
        
    def forward(self, dt=0.01, train=False, record=False):
        if record:
            self.history.append(self.r.cpu().clone())
            
    def __repr__(self):
        return f"SensoryBundle(out: {self.n_neurons}, t: {self.transmitter.data.cpu().numpy()})"

        
class OjaBundle(torch.nn.Module):

    def __init__(self, n_neurons, transmitter=[1, 0, 0, 0], thres=0.0, tau_v=0.5, tau_w=[5, 5, 5, 5], alpha=[0.25, 0.25, 0.25, 0.25], activation=Tanh(alpha=1.0), n_inputs=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.transmitter = torch.nn.Parameter(torch.tensor(transmitter, dtype=torch.float32), requires_grad=False)
        self.thres = thres
        self.tau_v = tau_v
        self.tau_w = torch.nn.Parameter(torch.tensor(tau_w, dtype=torch.float32), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.activation = activation
        
        self.v = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # voltage
        self.r = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # rate
        
        if n_inputs is not None:
            self.compile(n_inputs)

        self.history = History()
        self.compiled = False

    def compile(self, n_inputs):
        self.n_inputs = n_inputs
        self.x = torch.nn.Parameter(torch.zeros([n_inputs, 4]), requires_grad=False)
        self.w = torch.nn.Parameter(2 * torch.rand([self.n_neurons, n_inputs, 4]) * torch.sqrt(self.alpha**2 / n_inputs), requires_grad=False)
        self.compiled = True
        
    def rt(self):
        return torch.outer(self.r, self.transmitter)

    def forward(self, dt=0.01, train=False, record=False):
        I = (self.w * self.x).sum(1)
        
        if train:
            self.w[:] += (self.r.reshape(self.n_neurons, 1, 1) * ((self.x * self.alpha**2).reshape(1, self.n_inputs, 4) - self.w * I.reshape(self.n_neurons, 1, 4))) / self.tau_w * dt
            
        self.v[:] += (I[:,0] - I[:,1] + I[:,2] - I[:,3] - self.v) / self.tau_v * dt
        self.r[:] = self.activation(self.v - self.thres)
            
        if record:
            self.history.append(self.v.cpu().clone())
            
    def __repr__(self):
        if self.compiled:
            return f"Bundle(in: {self.n_inputs}, out: {self.n_neurons}, t: {self.transmitter.data.cpu().numpy()})"
        else:
            return f"Bundle( - , out: {self.n_neurons})"

        
class BCMBundle(torch.nn.Module):

    def __init__(self, n_neurons, transmitter=[1, 0, 0, 0], thres=0.0, tau_v=0.2, tau_t=1, tau_w=[10, 10, 10, 10], activation=Tanh(alpha=1.0), n_inputs=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.transmitter = torch.nn.Parameter(torch.tensor(transmitter, dtype=torch.float32), requires_grad=False)
        self.thres = thres
        self.tau_v = tau_v
        self.tau_t = tau_t
        self.tau_w = torch.nn.Parameter(torch.tensor(tau_w, dtype=torch.float32), requires_grad=False)
        self.activation = activation
        
        self.v = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # voltage
        self.t = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # bcm threshold
        self.r = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # rate
        
        if n_inputs is not None:
            self.compile(n_inputs)

        self.history = History()
        self.compiled = False

    def compile(self, n_inputs):
        self.n_inputs = n_inputs
        self.x = torch.nn.Parameter(torch.zeros([n_inputs, 4]), requires_grad=False)
        self.w = torch.nn.Parameter(torch.rand([self.n_neurons, n_inputs, 4]) / math.sqrt(n_inputs), requires_grad=False)
        self.compiled = True
        
    def rt(self):
        return torch.outer(self.r, self.transmitter)

    def forward(self, dt=0.01, train=False, record=False):
        I = (self.w * self.x).sum(1)
        
        if train:
            self.t[:] += (self.r - self.t) / self.tau_t * dt
            regularizer = (self.r * (self.r - self.t)).reshape(self.n_neurons, 1, 1)
            self.w[:] += (regularizer * self.x.reshape(1, self.n_inputs, 4)) / self.tau_w * dt
            
        self.v[:] += (I[:,0] - I[:,1] + I[:,2] - I[:,3] - self.v) / self.tau_v * dt
        self.r[:] = self.activation(self.v - self.thres)
            
        if record:
            self.history.append(self.v.cpu().clone())
            
    def __repr__(self):
        if self.compiled:
            return f"Bundle(in: {self.n_inputs}, out: {self.n_neurons}, t: {self.transmitter.data.cpu().numpy()})"
        else:
            return f"Bundle( - , out: {self.n_neurons})"
        
        
class LOLBundle(torch.nn.Module):
    """Combining Oja's rule and BCM rule."""
    def __init__(self, n_neurons, transmitter=[1, 0, 0, 0], thres=0.0, tau_v=0.2, tau_t=1, tau_w=[10, 10, 10, 10], alpha=[0.5, 0.5, 0.5, 0.5], activation=Tanh(alpha=1.0), n_inputs=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.transmitter = torch.nn.Parameter(torch.tensor(transmitter, dtype=torch.float32), requires_grad=False)
        self.thres = thres
        self.tau_v = tau_v
        self.tau_t = tau_t
        self.tau_w = torch.nn.Parameter(torch.tensor(tau_w, dtype=torch.float32), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.activation = activation
        
        self.v = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # voltage
        self.t = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # bcm threshold
        self.r = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # rate
        
        if n_inputs is not None:
            self.compile(n_inputs)

        self.history = History()
        self.compiled = False

    def compile(self, n_inputs):
        self.n_inputs = n_inputs
        self.x = torch.nn.Parameter(torch.zeros([n_inputs, 4]), requires_grad=False)
        self.w = torch.nn.Parameter(2 * torch.rand([self.n_neurons, n_inputs, 4]) * torch.sqrt(self.alpha**2 / n_inputs), requires_grad=False)
        self.compiled = True
        
    def rt(self):
        return torch.outer(self.r, self.transmitter)

    def forward(self, dt=0.01, train=False, record=False):
        I = (self.w * self.x).sum(1)
        
        if train:
            self.t[:] += (self.r - self.t) / self.tau_t * dt
            regularizer = (self.r * (self.r - self.t)).reshape(self.n_neurons, 1, 1)
            self.w[:] += (regularizer * ((self.x * self.alpha).reshape(1, self.n_inputs, 4) - self.w * I.reshape(self.n_neurons, 1, 4))) / self.tau_w * dt
            
        self.v[:] += (I[:,0] - I[:,1] + I[:,2] - I[:,3] - self.v) / self.tau_v * dt
        self.r[:] = self.activation(self.v - self.thres)
            
        if record:
            self.history.append(self.v.cpu().clone())
            
    def __repr__(self):
        if self.compiled:
            return f"Bundle(in: {self.n_inputs}, out: {self.n_neurons}, t: {self.transmitter.data.cpu().numpy()})"
        else:
            return f"Bundle( - , out: {self.n_neurons})"

        
class DVBundle(torch.nn.Module):

    def __init__(self, n_neurons, transmitter, thres=0.0, tau_v=0.2, tau_w=[50, 50, 50, 50], alpha=[0.5, 0.5, 0.5, 0.5], activation=Tanh(alpha=1.0), n_inputs=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.transmitter = torch.nn.Parameter(torch.tensor(transmitter, dtype=torch.float32), requires_grad=False)
        self.thres = thres
        self.tau_v = tau_v
        self.tau_w = torch.nn.Parameter(torch.tensor(tau_w, dtype=torch.float32), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.activation = activation
        
        self.v = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # voltage
        self.r = torch.nn.Parameter(torch.zeros([n_neurons]), requires_grad=False) # rate
        
        if n_inputs is not None:
            self.compile(n_inputs)

        self.history = History()
        self.compiled = False

    def compile(self, n_inputs):
        self.n_inputs = n_inputs
        self.x = torch.nn.Parameter(torch.zeros([n_inputs, 4]), requires_grad=False)
        self.w = torch.nn.Parameter(2 * torch.rand([self.n_neurons, n_inputs, 4]) * torch.sqrt(self.alpha**2 / n_inputs), requires_grad=False)
        self.compiled = True
        
    def rt(self):
        return torch.outer(self.r, self.transmitter)

    def forward(self, dt=0.01, train=False, record=False):
        I = (self.w * self.x).sum(1)
        dv = (I[:,0] - I[:,1] + I[:,2] - I[:,3] - self.v) / self.tau_v * dt
        
        if train:
            regularizer = (self.r * self.activation(self.v - self.thres, d=True) * dv).reshape(self.n_neurons, 1, 1)
            # regularizer = (self.activation(self.v - self.thres, d=True) * dv).reshape(self.n_neurons, 1, 1)
            # regularizer = (self.r * dv).reshape(self.n_neurons, 1, 1)
            self.w[:] += regularizer * ((self.x * self.alpha).reshape(1, self.n_inputs, 4) - self.w * I.reshape(self.n_neurons, 1, 4)) / self.tau_w
        
        self.r[:] = self.activation(self.v - self.thres)
        self.v[:] += dv
        
        if record:
            self.history.append(self.v.cpu().clone())
            
    def __repr__(self):
        if self.compiled:
            return f"Bundle(in: {self.n_inputs}, out: {self.n_neurons}, t: {self.transmitter.data.cpu().numpy()})"
        else:
            return f"Bundle( - , out: {self.n_neurons})"
        

class Network(torch.nn.Module):

    def __init__(self, device="cpu"):
        super().__init__()
        self.bundles = torch.nn.ModuleDict()
        self.connections = OrderedDict()
        
        self.device = device

    def add(self, name, bundle):
        if name not in self.bundles:
            self.bundles[name] = bundle
        else:
            print("bundle '{name}' already exists.")

    def connect(self, tail, head):
        if head in self.connections:
            self.connections[head] += [tail]
        else:
            self.connections[head] = [tail]

    def compile(self):
        for head in self.connections:
            n_inputs = sum([self.bundles[tail].n_neurons for tail in self.connections[head]])
            self.bundles[head].compile(n_inputs)
            
        self.to(self.device)

    def forward(self, dt=0.01, train=False, record=False):
        for head in self.connections:
            self.bundles[head].x[:] = torch.cat([self.bundles[tail].rt() for tail in self.connections[head]])
            
        for name in self.bundles:
            self.bundles[name].forward(dt, train=train, record=record)
        
#         threads = []
#         for name in self.bundles:
#             thread = threading.Thread(target=self.bundles[name].forward, kwargs={"dt": dt, "train": train, "record": record})
#             thread.start()
            
#         for thread in threads:
#             thread.join()
                
    def reset(self):
        for name in self.bundles:
            self.bundles[name].history.reset()
        