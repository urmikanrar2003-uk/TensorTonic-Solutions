import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x=np.array(x)
    p=np.array(p)
    s=np.sum(p)
    if (s==1 and x.shape==p.shape):
        Ex=float(np.sum(np.multiply(x,p)))
        return Ex
    else:
        raise ValueError("sum of probabilities is not equal 1 or p and x is not same size")