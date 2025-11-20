def getVersionNumber():
    try:
        from importlib.metadata import version
    except ImportError:
        # Fallback for Python < 3.8
        from importlib_metadata import version
    return version("SFcalculator_torch")

__author__ = "Minhuan Li"
__email__ = "minhuanli41@gmail.com"
__version__ = getVersionNumber()

# Lazy loading to speed up imports while maintaining API compatibility
_LAZY_IMPORTS = {
    'SFcalculator': 'Fmodel',
    'PDBParser': 'io',
    'fetch_pdb': 'io',
    'fetch_pdbredo': 'io',
    'get_polar_axis': 'symmetry',
    'utils': 'utils',
    'patterson': 'patterson',
    'packingscore': 'packingscore',
}

def __getattr__(name):
    """Lazy import mechanism for top-level API"""
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        import importlib
        
        # Import the module
        if name in ['utils', 'patterson', 'packingscore']:
            # These are submodule imports
            module = importlib.import_module(f'.{module_name}', __name__)
            globals()[name] = module
            return module
        else:
            # These are specific objects from modules
            module = importlib.import_module(f'.{module_name}', __name__)
            obj = getattr(module, name)
            globals()[name] = obj
            return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """Support for dir() to show available lazy imports"""
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())


