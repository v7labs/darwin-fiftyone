"""darwin-fiftyone package.

Keep imports lightweight at package import time.

`darwin_fiftyone.darwin` imports the Darwin SDK and FiftyOne, which are
heavyweight dependencies and may not be available in all environments (for
example, when running lightweight unit tests).
"""

__all__ = ["DarwinBackendConfig", "DarwinBackend"]


def __getattr__(name: str):
    if name in __all__:
        # Import lazily to avoid importing Darwin SDK / FiftyOne unless needed
        from .darwin import DarwinBackend, DarwinBackendConfig

        return {
            "DarwinBackendConfig": DarwinBackendConfig,
            "DarwinBackend": DarwinBackend,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
