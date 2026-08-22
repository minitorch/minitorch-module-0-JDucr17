"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        a: First input value.
        b: Second input value.

    Returns:
    -------
        Product of ``a`` and ``b``.

    """
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
    ----
        a: First input value.
        b: Second input value.

    Returns:
    -------
        Sum of ``a`` and ``b``.

    """
    return a + b


def id(x: float) -> float:
    """Return the input value unchanged.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The unchanged input value.

    """
    return x


def neg(x: float) -> float:
    """Negate an input value.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        Negation of ``x``.

    """
    return -x


def lt(a: float, b: float) -> float:
    """Check whether one value is less than another.

    Args:
    ----
        a: First input value.
        b: Second input value.

    Returns:
    -------
        1.0 if ``a`` is less than ``b``, otherwise 0.0.

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Check whether two values are equal.

    Args:
    ----
        a: First input value.
        b: Second input value.

    Returns:
    -------
        1.0 if ``a`` and ``b`` are equal, otherwise 0.0.

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Return the larger of two values.

    Args:
    ----
        a: First input value.
        b: Second input value.

    Returns:
    -------
        Larger of ``a`` and ``b``.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check whether two values are close to each other.

    Args:
    ----
        a: First input value.
        b: Second input value.

    Returns:
    -------
        True if the absolute difference is less than 1e-2, otherwise False.

    """
    return abs(a - b) < 1e-2


def exp(x: float) -> float:
    """Compute the exponential of an input value.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        Exponential of ``x``.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal of an input value.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        Reciprocal of ``x``.

    """
    return 1.0 / x


def relu(x: float) -> float:
    """Apply the ReLU activation function to an input value.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        ``x`` if it is positive, otherwise 0.0.

    """
    return max(0.0, x)


def sigmoid(x: float) -> float:
    """Apply the sigmoid activation function to an input value.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        float between 0 and 1.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))

    return math.exp(x) / (1.0 + math.exp(x))


def log(x: float) -> float:
    """Log function of x"""
    return math.log(x)


def log_back(x: float, arg: float) -> float:
    """Computes the derivative of natural logarithm of x and multiplies it with arg

    Args:
    ----
        x: Input value
        arg: Input value, multiplication factor

    Returns:
    -------
        derivative of natural log of x multiplied by arg

    """
    return 1 / x * arg


def inv_back(x: float, arg: float) -> float:
    """Computes the derivative of the reciprocal function and multiplies it by arg

    Args:
    ----
        x: Input value
        arg: Input value, multiplication factor

    Returns:
    -------
        derivative of the reciprocal of x multiplied by arg

    """
    return (-1 / x**2) * arg


def relu_back(x: float, arg: float) -> float:
    """Computes the derivative of the ReLU function and multiplies it by arg

    Args:
    ----
        x: Input value
        arg: Input value, multiplication factor

    Returns:
    -------
        derivative of the ReLu function of x multiplied by arg

    """
    if x <= 0:
        return 0.0

    return arg


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order function that applies function over each element of a collection"""

    def apply(values: Iterable[float]) -> Iterable[float]:
        ret = []
        for element in values:
            res = fn(element)
            ret.append(res)

        return ret

    return apply


negList = map(lambda x: -x)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher order function that applies function over corresponding elements of two lists"""

    def apply(first: Iterable[float], second: Iterable[float]) -> Iterable[float]:
        ret = []

        for x, y in zip(first, second):
            res = fn(x, y)
            ret.append(res)

        return ret

    return apply


addLists = zipWith(add)


def reduce(
    fn: Callable[[float, float], float],
    start: float,
) -> Callable[[Iterable[float]], float]:
    """Higher order function that transforms a collection of data into a single cumulative result"""

    def apply(values: Iterable[float]) -> float:
        acc = start
        for value in values:
            acc = fn(value, acc)
        return acc

    return apply


sum = reduce(add, 0)
prod = reduce(lambda x, y: x * y, 1)


# TODO: Implement for Task 0.3.
