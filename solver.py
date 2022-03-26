import numbers

import numpy as np


class EquationSolver:
    def __init__(
        self,
        solver="bisect",
        max_iter=10000,
        interval=(-10000, 10000),
        precision=1e-10,
        n_sample=100000,
        random_state=None,
    ) -> None:
        """
        If possible, set `interval` to `(a, b)` satisfying f(a)f(b) < 0.
        """
        all_solvers = {"bisect", "secant", "golden"}
        if solver not in all_solvers:
            raise ValueError("Equation Solver supports only solvers in {}, got {}.".format(all_solvers, solver))
        self.__solver = solver

        if solver == "secant" and not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter parameter expects integer greater than 0, got {}.".format(max_iter))
        self.__max_iter = max_iter

        try:
            iter(interval)
            if not all(map(isinstance, interval, [numbers.Number, numbers.Number])):
                raise ValueError(
                    "interval parameter expects Iterable[Number, Number], got {}{}".format(
                        type(interval),
                        list(map(type, interval))
                    )
                )
        except TypeError:
            raise ValueError(
                "interval parameter expects Iterable[Number, Number], got {}".format(
                    type(interval)
                )
            )
        self.__interval_max, self.__interval_min = sorted(interval)

        if solver in {"bisect", "golden"} and not isinstance(precision, numbers.Number) or precision <= 0:
            raise ValueError("precision parameter expects unsigned number, got {}.".format(precision))
        self.__precision = precision

        if not isinstance(n_sample, numbers.Integral) or n_sample <= 0:
            raise ValueError("n_sample parameter expects integer greater than 0, got {}.".format(n_sample))
        self.__n_sample = n_sample

        if random_state is not None:
            if not isinstance(random_state, int) or random_state < 0 or (2**32 - 1) < random_state:
                raise ValueError("random_state parameter expects integer between 0 and 2**32 - 1, got {}.".format(random_state))
            np.random.seed(random_state)

    def solve(self, func):
        """
        Solves the equation `f(x) = 0` where f is the given function in the parameter `func`.
        If the function is guaranteed to be continuous and RuntimeError is raised, try narrowing the `interval`.

        Parameter
        ---
        `func`: Equation to solve. Functions must be continuous.

        Return
        ---
        One of the solutions to the equation.
        """
        self.__func = np.vectorize(func)
        self.__setInterval()
        if self.__solver == "bisect":
            return self.__binary_search()
        elif self.__solver == "secant":
            return self.__secant()
        elif self.__solver == "golden":
            return self.__golden()

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

    def __secant(self):
        previous, current = self.__interval_min, self.__interval_max
        for _ in range(self.__max_iter):
            if previous == current:
                break
            previous, current = current, current - self.__func(current) * (current - previous) / (self.__func(current) - self.__func(previous))
        return current

    def __golden(self):
        gamma = 0.3819660112501051
        f = self.__func
        a = self.__interval_min
        b = self.__interval_max
        a_prev = b_prev = None
        while (b - a) > self.__precision:
            if a == b or (a == a_prev and b == b_prev):
                break
            a_prev = a
            b_prev = b
            x1 = a + gamma * (b - a)
            x2 = a + (1 - gamma) * (b - a)
            if np.sign(f(a)) == np.sign(f(x1)):
                a = x1
                x1 = x2
                x2 = a + (1 - gamma) * (b - a)
            else:
                b = x2
                x2 = x1
                x1 = a + gamma * (b - a)
        return x1
