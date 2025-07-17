from dynamic_profiler import dynamic_profiler


class A:
    def __init__(self, name: str):
        self.name = name

    @dynamic_profiler
    def count_fibonacci(self, n: int) -> int:
        self.fib_line = [0, 1, 1]
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n in range(len(self.fib_line)):
            return self.fib_line[n]
        else:
            for i in range(2, n + 1):
                next_fib = self.fib_line[-1] + self.fib_line[-2]
                self.fib_line.append(next_fib)
        return self.fib_line[n]


if __name__ == "__main__":
    a = A("Fibonacci Calculator")
    print(f"The 11th Fibonacci number is: {a.count_fibonacci(n=11)}")
