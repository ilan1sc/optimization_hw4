
import numpy as np
from constrained_min import interior_pt
from examples import test_qp
from examples import test_lp
from utils import plot_results_linear
from utils import plot_results_quadratic

print('Select a function for analysis:')
print('1 - Quadratic')
print('2 - Linear')
selected_function = input('Enter a number 1 / 2:')
selected_function = int(selected_function)

# quadratic function
if selected_function == 1:
    func_name = 'quadratic'
    backtrack_flag = True
    func_min = test_qp
    start_x = np.array([0.1, 0.2, 0.7], dtype=np.float64)
    newton_outcome = interior_pt(func_min, start_x, backtrack=backtrack_flag, m=4, t=1.0, miu=10, eps_barrier=1e-5,
                                 eps_newton=1e-5)

    # Plot results
    x_trajectory = newton_outcome[2]
    f_trajectory = newton_outcome[3]
    iteration_range = np.arange(len(f_trajectory))
    x_limits = np.array([-2, 2])
    y_limits = np.array([-2, 2])
    z_limits = np.array([-2, 2])
    plot_results_quadratic(func_min, x_trajectory, f_trajectory, x_limits, y_limits, z_limits, func_name)

elif selected_function == 2:

    func_name = 'linear'
    backtrack_flag = False
    func_min = test_lp
    start_x = np.array([0.5, 0.75], dtype=np.float64)
    newton_outcome = interior_pt(func_min, start_x, backtrack=backtrack_flag, m=4, t=1.0, miu=10, eps_barrier=1e-5,
                                 eps_newton=1e-5)

    # Plot results
    x_trajectory = newton_outcome[2]
    f_trajectory = newton_outcome[3]
    x_limits = np.array([-1, 3])
    y_limits = np.array([-1, 3])
    plot_results_linear(func_min, x_trajectory, f_trajectory, x_limits, y_limits, func_name)

else:
    print("select  1 or 2"
          .format(function_index))




