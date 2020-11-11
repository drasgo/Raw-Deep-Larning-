import pprint


class Verbosity:
    """ """
    DEBUG = True
    RELEASE = False


def logger(message, verbose: Verbosity = Verbosity.RELEASE):
    """

    :param message: 
    :param verbose: Verbosity:  (Default value = Verbosity.RELEASE)

    """
    if verbose is Verbosity.DEBUG:
        if isinstance(message, str) or isinstance(message, int):
            print("\n________________")
            print(message)
            # input()
        else:
            pprint.pprint(message)
