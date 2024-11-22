pso_params = {
    'alpha': 0.9,        # inertia weight - not really used as inertia decay happens in PSO.py
    'beta': 2.0,         # cognitive weight
    'gamma': 2.0,        # social weight (weight of informants)
    'delta': 1.5,        # global weight
    'epsilon': 0.005,     # step size
    'v_max_factor': 0.05  # velocity max
}
reset_parameters = {
    'min_spread': 0.5,
    'max_spread': 2.0,      #bounds
    'velocity_range': (-0.1, 0.1),
    'improvement_threshold': 1e-6
}
update_parameters = {
    'update_frequency': 10  # Update informants every 20 iterations
}