from __future__ import annotations




class Registry:
    """Maintain a mapping from names to callables or classes."""
    def __init__(self, name):
        self._name = name
        self._dict = {}
    def register(self, key=None):
        print('a')
        def decorator(obj):
            name = key or obj.__name__
            if name in self._dict:
                raise KeyError(f"{name} already registered in {self._name}")
            self._dict[name.lower()] = obj
            return obj
        return decorator
    def get(self, name):
        if name.lower() not in self._dict:
            raise KeyError(f"{name} not found in {self._name}")
        return self._dict[name.lower()]
    def list(self):
        return list(self._dict.keys())
    def has(self, name):
        print(self._dict)
        print(name)
        return name.lower() in self._dict