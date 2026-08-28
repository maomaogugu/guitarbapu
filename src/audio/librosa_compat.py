"""Compatibility helpers for importing librosa across supported Python versions."""

import sys


def import_librosa():
    """Import librosa with a Python 3.14 numba-cache workaround."""

    if sys.version_info >= (3, 14):
        import numba.core.caching as numba_caching
        import numba.core.dispatcher as numba_dispatcher
        import numba.np.ufunc.ufuncbuilder as numba_ufuncbuilder
        import numba.np.ufunc.wrappers as numba_wrappers

        numba_dispatcher.FunctionCache = lambda function: numba_caching.NullCache()
        numba_ufuncbuilder.FunctionCache = (
            lambda function: numba_caching.NullCache()
        )
        numba_wrappers.GufWrapperCache = lambda **kwargs: numba_caching.NullCache()

    import librosa

    return librosa


__all__ = ["import_librosa"]
