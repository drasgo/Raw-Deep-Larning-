class Verbosity:
    DEBUG = True
    RELEASE = False


def logger(message: str, verbose: Verbosity = Verbosity.RELEASE):
    if verbose is Verbosity.DEBUG:
        print(message)
