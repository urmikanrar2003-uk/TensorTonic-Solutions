def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    def diff_x(x):
        return 2*a*x + b
    x=float(x0)
    for i in range(steps):
        x=x-lr*diff_x(x)
    return float(x)