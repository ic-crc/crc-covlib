# Numba's JIT on-disk caching
# If any problem have been introduced from using on-disk caching, you may want to
# delete the 'crc_covlib/helper/__pycache__' folder to clear the cache.
COVLIB_NUMBA_CACHE=False # not using on-disk caching by default

try:
    from numba import config
    config.DISABLE_JIT = True # Numba JIT is disabled by default
    from numba import jit
    #print('Numba is installed, DISABLE_JIT={}'.format(config.DISABLE_JIT))
except:
    # Do-nothing jit decorator so that jitted covlib functions may still be
    # called when numba is not installed or failed to be loaded (conflict
    # with other packages for example).
    def jit(nopython, cache=False):
        def decorator(func):
            return func
        return decorator
    #print('Numba is not installed or it failed to be loaded')


# Call before any helper module is loaded to use Numba JIT on supported functions.
def EnableNumbaJIT(onDiskCaching: bool=False) -> None:
    from numba import config
    config.DISABLE_JIT = False
    global COVLIB_NUMBA_CACHE
    COVLIB_NUMBA_CACHE = onDiskCaching