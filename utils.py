

import numpy as np
import matplotlib.pyplot as plt


def plot_linear_feasible_set(x_values, y_values):
    constraint1 = lambda x, y: - x - y + 1
    constraint2 = lambda x, y: y - 1
    constraint3 = lambda x, y: x - 2
    constraint4 = lambda x, y: -y

    constraints = (constraint1(x_values, y_values) <= 0) & \
                  (constraint2(x_values, y_values) <= 0) & \
                  (constraint3(x_values, y_values) <= 0) & \
                  (constraint4(x_values, y_values) <= 0)

    plt.imshow(
        constraints.astype(int),
        extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
        origin='lower',
        cmap='viridis'
    )


def plot_linear_contour(objective_function, limits_x, limits_y):

    # Create a meshgrid covering the relevant area
    x_range = np.linspace(limits_x[0], limits_x[1], 100)
    y_range = np.linspace(limits_y[0], limits_y[1], 100)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    rows, cols = grid_x.shape
    function_values = np.zeros((rows, cols), dtype=np.float64)

    # Calculate the function values for each point in the meshgrid
    for row in range(rows):
        for col in range(cols):
            input_values = np.array([grid_x[row, col], grid_y[row, col]], dtype=np.float64)
            function_value, _, _ = objective_function(input_values, t=0)
            function_values[row, col] = function_value

    # Create the contour plot
    contour = plt.contour(x_range, y_range, function_values)
    plt.clabel(contour, fmt='%1.2f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Constrains contours and path')
    plt.show()


def plot_learning_curve(function_values, function_name):

    fig, ax = plt.subplots()
    ax.plot(range(len(function_values)), function_values, color='b')
    ax.scatter(len(function_values) - 1, function_values[-1], color='r')
    ax.set_title('Log-barrier method for $f(x) = f$({})'.format(function_name))
    ax.set_xlabel('Nu of outer iterations')
    ax.set_ylabel('Function value ($f(x)$)')
    plt.show()


def plot_results_linear(minimization_function, track_x, track_f, limits_x, limits_y, function_name):

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(track_x[:, 0], track_x[:, 1], 'b')
    ax.scatter(track_x[-1][0], track_x[-1][1], color='r')

    x_range = np.linspace(limits_x[0], limits_x[1], 200)
    y_range = np.linspace(limits_y[0], limits_y[1], 200)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    plot_linear_feasible_set(grid_x, grid_y)
    plot_linear_contour(minimization_function, limits_x, limits_y)
    plot_learning_curve(track_f, function_name)


def plot_quadratic_feasible_set(x_values, y_values):

    constraint1 = lambda x, y: - x
    constraint2 = lambda x, y: - y

    constraints = (constraint1(x_values, y_values) <= 0) & \
                  (constraint2(x_values, y_values) <= 0)

    plt.imshow(
        constraints.astype(int),
        extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
        origin='lower',
        cmap='viridis'

    )


def plot_quadratic_contour(objective_function, limits_x, limits_y, limits_z):

    # Create a meshgrid covering the relevant area
    x_range = np.linspace(limits_x[0], limits_x[1], 100)
    y_range = np.linspace(limits_y[0], limits_y[1], 100)
    z_range = np.linspace(limits_z[0], limits_z[1], 100)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range)
    x_dim, y_dim, z_dim = grid_x.shape
    function_values = np.zeros((x_dim, y_dim, z_dim), dtype=np.float64)

    # Calculate the function values for each point in the meshgrid
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                input_values = np.array([grid_x[x, y, z], grid_y[x, y, z], grid_z[x, y, z]], dtype=np.float64)
                function_value, _, _ = objective_function(input_values, t=0)
                function_values[x, y, z] = function_value

    # Create the contour plot
    contour = plt.contour(x_range, y_range, function_values[:, :, 0])
    plt.clabel(contour, fmt='%1.2f')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Constrains contours and path')
    plt.show()


def plot_results_quadratic(minimization_function, track_x, track_f, limits_x, limits_y, limits_z,
                                function_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(track_x[:, 0], track_x[:, 1], 'b')
    ax.scatter(track_x[-1][0], track_x[-1][1], color='r')

    x_range = np.linspace(limits_x[0], limits_x[1], 100)
    y_range = np.linspace(limits_y[0], limits_y[1], 100)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    plot_quadratic_feasible_set(grid_x, grid_y)
    plot_quadratic_contour(minimization_function, limits_x, limits_y, limits_z)
    plot_learning_curve(track_f, function_name)

