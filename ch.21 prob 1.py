import numpy as np

def my_int_calc(f, f0, a: float, b: float, N: int, option: chr):
    'option = ‘rect’, ‘trap’, or ‘simp’'
    """a and b are scalars such that a < b, 
    N is a positive integer, and optionn
    is the string ‘rect’, ‘trap’, or ‘simp’. 
    Let x be an array starting at a, ending at b, 
    containing N evenly spaced elements.
    The output argument, I, should be an 
    approximation to the integral of f(x),
    with initial condition f0, computed
    according to the input argument, option."""

    h = (b - a) / (N - 1)
    x = np.linspace(a, b, N)

    if option == 'rect':
        I = h * sum(f((x[:N - 1] \
        + x[1:])/2))
    elif option == 'trap':
        I = (h/2)*(f(a) + \
                2 * sum(f(x[1:-1])) + f(b))
    elif option == 'simp':
        if N % 2 == 0:
            raise ValueError("Simpson's rule requires an odd number of points(N)")
        I = (h/3) * (f(a) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-2:2])) + f(b))
    else:
        raise ValueError("Option must be 'rect', 'trap', or 'simp'")
    return f0 + I
    



   