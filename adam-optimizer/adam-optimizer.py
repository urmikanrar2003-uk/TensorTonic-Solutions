import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param=np.array(param,dtype=float)
    grad=np.array(grad,dtype=float)
    m=np.array(m,dtype=float)
    v=np.array(v,dtype=float)
    m=beta1*m + (1-beta1)*grad
    v=beta2*v + (1-beta2)*grad*grad
    m_new=m/(1-(beta1**t))
    v_new=v/(1-(beta2**t))
    p=m_new/(np.sqrt(v_new)+eps)
    param_new=param-lr*p
    return (param_new.tolist(),m.tolist(),v.tolist())
    
    