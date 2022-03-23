import numpy as np


class EquationSolver:
    def __init__(
        self,
        interval=(-(2**31), (2**31 - 1)),
        precision=1e-10,
        n_sample=1e5,
        random_state=None,
    ) -> None:
        """
        If possible, set `interval` to `(a, b)` satisfying f(a)f(b) < 0.
        """
        self.__interval_max, self.__interval_min = sorted(interval)
        self.__precision = precision
        if n_sample <= 0:
            raise ValueError("`n_sample` must be greater than 0.")
        self.__n_sample = n_sample
        if random_state is not None:
            np.random.seed(random_state)

    def solve(self, func):
        """
        Solves the equation `f(x) = 0` where f is the given function in the paramater `func`.
        If the function is guaranteed to be continuous and RuntimeError is raised, try narrowing the `interval`.

        Paramater
        ---
        `func`: Equation to solve. Functions must be continuous.

        Return
        ---
        One of the solutions to the equation.
        """
        self.__func = np.vectorize(func)
        self.__setInterval()
        return self.__binary_search()

    def __setInterval(self) -> None:
        interval = [self.__interval_min, self.__interval_max]
        count_sample = 0
        while np.prod(np.sign(self.__func(interval))) >= 0:
            if count_sample == self.__n_sample:
                raise RuntimeError
            interval = (self.__interval_max - self.__interval_min) * np.random.rand(2) + self.__interval_min
            count_sample += 1
        interval.sort()
        self.__interval_min, self.__interval_max = interval

    def __binary_search(self):
        left = self.__interval_min
        right = self.__interval_max
        while (right - left) >= self.__precision:
            mid = left + (right - left) / 2
            if np.sign(self.__func(mid)) == np.sign(self.__func(left)):
                left = mid
            else:
                right = mid
        return left + (right - left) / 2
