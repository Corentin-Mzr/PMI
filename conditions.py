import numpy as np
from typing import Callable
from parameters import dt, length


def test_case_1d(test_id: int) -> tuple[Callable, ...]:
    """
    Define test cases for 1D

    u0: Value when x - ct > 0

    g: Value when x - ct < 0

    f: Source term

    :param test_id: Test case's id
    :return: Functions used for the test
    """
    def test_case_1() -> tuple[Callable, ...]:
        def u0(x: float) -> float:
            return x

        def g(t: float) -> float:
            return np.sin(5 * np.pi * t)

        def f(t: float, x: float) -> float:
            return 0

        return u0, g, f

    def test_case_2() -> tuple[Callable, ...]:
        def u0(x: float) -> float:
            return np.exp(-x**2)

        def g(t: float) -> float:
            return np.sin(5 * np.pi * t)

        def f(t: float, x: float) -> float:
            return 0

        return u0, g, f

    def test_case_3() -> tuple[Callable, ...]:
        def u0(x: float) -> float:
            return np.exp(-x**2)

        def g(t: float) -> float:
            return 4 * np.abs(np.sin(5 * np.pi * t))

        def f(t: float, x: float) -> float:
            return 2

        return u0, g, f

    def test_case_4() -> tuple[Callable, ...]:
        def u0(x: float) -> float:
            return x

        def g(t: float) -> float:
            return 4 * np.abs(np.sin(8 * np.pi * t))

        def f(t: float, x: float) -> float:
            return 2

        return u0, g, f

    def test_case_5() -> tuple[Callable, ...]:
        def u0(x: float) -> float:
            return x

        def g(t: float) -> float:
            return 4 * np.abs(np.sin(8 * np.pi * t))

        def f(t: float, x: float) -> float:
            return 2

        return u0, g, f

    test_cases = {1: test_case_1(),
                  2: test_case_2(),
                  3: test_case_3(),
                  4: test_case_4(),
                  5: test_case_5()}

    if test_id in test_cases.keys():
        return test_cases.get(test_id)
    raise KeyError(f'No such test case: {test_id}')


def vectorize_test_case(test_case: tuple[Callable, ...]) -> tuple[Callable, ...]:
    """
    :param test_case: u0, g and f
    :return: Vectorized version of the test case functions
    """
    u0, g, f = test_case
    u0_vec = np.vectorize(u0, otypes=[float])
    g_vec = np.vectorize(g, otypes=[float])
    f_vec = np.vectorize(f, otypes=[float])

    return u0_vec, g_vec, f_vec


def test_case_2d(test_id: int) -> tuple[Callable, ...]:
    """
    Define test cases for 2D

    u0: Value when ||(x,y)||² - ct > 0

    g: Value when ||(x,y)||² - ct - ct < 0

    f: Source term

    :param test_id: Test case's id
    :return: Functions used for the test
    """
    def test_case_1() -> tuple[Callable, ...]:
        def u0(x: float, y: float) -> float:
            return x + y

        def g(t: float) -> float:
            return np.sin(5 * np.pi * t)

        def f(t: float, x: float, y: float) -> float:
            return 0

        return u0, g, f

    def test_case_2() -> tuple[Callable, ...]:
        def u0(x: float, y: float) -> float:
            return np.exp(-x**2 - y**2)

        def g(t: float) -> float:
            return np.sin(5 * np.pi * t)

        def f(t: float, x: float, y: float) -> float:
            return 0

        return u0, g, f

    def test_case_3() -> tuple[Callable, ...]:
        def u0(x: float, y: float) -> float:
            return np.exp(-x**2 - y**2)

        def g(t: float) -> float:
            return 4 * np.abs(np.sin(5 * np.pi * t))

        def f(t: float, x: float, y: float) -> float:
            return 2

        return u0, g, f

    def test_case_4() -> tuple[Callable, ...]:
        def u0(x: float, y: float) -> float:
            return 0

        def g(t: float) -> float | np.ndarray:
            return np.where(t < 2 * dt, np.abs(np.sin(10 * np.pi * t)), 0)

        def f(t: float, x: float, y: float) -> float:
            return 0

        return u0, g, f

    test_cases = {1: test_case_1(),
                  2: test_case_2(),
                  3: test_case_3(),
                  4: test_case_4()}

    if test_id in test_cases.keys():
        return test_cases.get(test_id)
    raise KeyError(f'No such test case: {test_id}')
