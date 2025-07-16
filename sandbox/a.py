class A:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def count_fibonacci(self, n: int) -> int:
        """Calculate the nth Fibonacci number."""
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
    print(f"The 11th Fibonacci number is: {a.count_fibonacci(11)}")
    print(f"Fibonacci sequence up to 11: {a.fib_line}")