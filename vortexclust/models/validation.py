def check_unfitted_model(model) -> None:
    r"""
    Checks if a clustering model is unfitted and has required attributes.

    :param model: A clustering model instance to be validated.
    :type model: object

    :raises TypeError: If the model is already fitted or doesn't implement a callable fit_predict method.
    :raises ValueError: If the model doesn't support the 'n_clusters' parameter.

    :return: None
    """
    # check if model is fitted, does not support n_clusters or fit_predict
    if hasattr(model, 'labels_') or hasattr(model, 'fit_predict_called'):
        raise TypeError("Passed model appears to be fitted. Please provide an unfitted model instance.")
    if not hasattr(model, 'fit_predict') or not callable(getattr(model, 'fit_predict')):
        raise AttributeError("Provided model must implement a callable 'fit_predict(X)' method.")
    if 'n_clusters' not in model.get_params():
        raise ValueError("Provided model must support the 'n_clusters' parameter via set_params().")
