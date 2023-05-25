import math

class PowerConvertor:
    def __init__(self, base=2, dtype=int, *, default):
        self._default = default
        self.base = base
        self.dtype = dtype

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value: int):
        setattr(obj, self._name, self.base**self.dtype(value))


class IntegerConvertor:
    def __init__(self, *, default):
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value: int):
        setattr(obj, self._name, int(value))


class LambdaConvertor:
    def __init__(self, func, *, default):
        self._func = func
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value: int):
        setattr(obj, self._name, self._func(value))