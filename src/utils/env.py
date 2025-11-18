def is_colab():
    """
    Detect if running inside Google Colab.
    Returns:
        bool: True if Google Colab environment is detected.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False
