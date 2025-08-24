def validate_input(data):
    if not isinstance(data, (list, dict)):
        raise ValueError("Input data must be a list or a dictionary.")
    # Additional validation logic can be added here

def validate_model_params(params):
    required_params = ['order', 'seasonal_order']
    for param in required_params:
        if param not in params:
            raise ValueError(f"Missing required model parameter: {param}")
    # Additional validation logic can be added here