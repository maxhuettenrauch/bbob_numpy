import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from objective_functions.f_rosenbrock import Rosenbrock


if __name__ == '__main__':
    # check two dimensional rosenbrock function
    plt.interactive(True)
    r = Rosenbrock(2)

    nx, ny = (101, 101)
    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1, 2, ny)
    xv, yv = np.meshgrid(x, y)

    data = np.stack((xv.flatten(), yv.flatten()), axis=0)

    out = r.function_eval(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')

    ax.contourf(xv, yv, out.reshape([nx, ny]))
    # plt.colorbar()
    plt.show(block=True)

    print(np.min(out), data[:, np.argmin(out)])
