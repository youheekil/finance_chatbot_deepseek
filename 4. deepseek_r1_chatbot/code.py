def fibonacci(n):
    if n < 0:
        print("Invalid input: Negative number")
        return None
    elif n == 0 or n == 1:
        print(f"Fibonacci({n}) = {n}")
        return n
    else:
        a, b = 0, 1
        for i in range(2, n + 1):
            next_num = a + b
            print(f"Calculating Fibonacci({i}):")
            print(f"a={a}, b={b}")
            print(f"Next number: {next_num}")
            a, b = b, next_num
        print(f"Fibonacci({n}) = {b}")
        return b

# Example usage:
print("Fibonacci sequence up to 10:")
for i in range(0, 11):
    print(fibonacci(i))