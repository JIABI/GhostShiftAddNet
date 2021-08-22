import torch
from torch.optim.optimizer import Optimizer, required


class SVRG(Optimizer):
    r""" implement SVRG """

    def __init__(self, params, lr=required, freq=3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, freq=freq,momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.counter = 0
        self.counter2 = 0
        self.flag = False
        super(SVRG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq = group['freq']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # p.data.add_(-group['lr'], d_p)

                if 'large_batch' not in param_state:
                    buf = param_state['large_batch'] = torch.zeros_like(p.data)
                    buf.add_(d_p)  # add first large, low variance batch
                    # need to add the second term in the step equation; the gradient for the original step!
                    buf2 = param_state['small_batch'] = torch.zeros_like(p.data)

                buf = param_state['large_batch']
                buf2 = param_state['small_batch']

                if self.counter != freq:
                    buf.data = d_p.clone()  # copy new large batch. Begining of new inner loop
                    temp = torch.zeros_like(p.data)
                    buf2.data = temp.clone()
                if self.counter2 == 0:
                    buf.data += d_p.clone()  # copy new large batch. Begining of new inner loop
                    buf.data = buf.data / len(d_p)
                    buf2.data.add_(d_p)  # first small batch gradient for inner loop!

                # dont update parameters when computing large batch (low variance gradients)
                #if self.counter == freq and self.flag != False:
                p.data.add_(-group['lr'],(0.375*(d_p - buf2) + 0.625*buf))
                #elif self.counter != freq:
                #    p.data.add_(-group['lr'], (0.375* (d_p - buf2)))

                param_state = self.state[p]

        self.flag = True  # rough way of not updating the weights the FIRST time we calculate the large batch gradient

        if self.counter == freq:
            self.counter = 0
            self.counter2 = 0

        self.counter += 1
        self.counter2 += 1

        return loss

    def get_d(self, param):
        """Performs a single optimization step.

        Arguments:
            param (parameter, optional): the parameter whose update is calculated.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p is param:
                    if param.grad is None:
                        return None
                    d_p = param.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, param.data)
                    if momentum != 0:
                        param_state = self.state[param]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    return -group['lr'] * d_p

        return None