import numpy as np
import matplotlib.pyplot as plt
from objective_functions.f_rosenbrock import Rosenbrock


if __name__ == '__main__':
    # check two dimensional rosenbrock function
    plt.interactive(True)
    r = Rosenbrock(2)

    nx, ny = (101, 101)
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    xv, yv = np.meshgrid(x, y)

    data = np.stack((xv.flatten(), yv.flatten()), axis=1)

    out = r.evaluate_full(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.contour(xv, yv, np.log(out.reshape([nx, ny])))
    # plt.colorbar()
    plt.show(block=True)

    print(np.min(out), data[np.argmin(out), :])
