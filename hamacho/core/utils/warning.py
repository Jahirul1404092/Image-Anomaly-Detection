import warnings


def hide_warns():
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    warnings.filterwarnings("ignore")
