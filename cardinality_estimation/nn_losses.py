

class NNLoss():

    def __init__(self, training_samples, **kwargs):
        pass

    def compute_loss(self, net):
        '''
        @net: torch based neural net.

        How the training samples are selected etc. is entirely upto the
        implementation for this class.

        @ret: torch tensor, loss, which will be used to do gradient descent.
        '''
        pass
