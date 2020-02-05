class Mutan_OPT:
    def __init__(self, dim_v, dim_q, dim_hq, dim_hv, dim_mm):
        self.dim_v = dim_v
        self.dim_q = dim_q
        self.dim_hq = dim_hq
        self.dim_hv = dim_hv
        self.dim_mm = dim_mm
        self.dropout_q = 0.5
        self.dropout_v = 0.5
        self.dropout_hq = 0
        self.dropout_hv = 0
        self.activation_q = 'tanh'
        self.activation_v = 'tanh'
        self.activation_hq = None
        self.activation_hv = None
        self.activation_mm = None
        self.R = 10
