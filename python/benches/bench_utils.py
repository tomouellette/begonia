import timeit
import functools


def runtime_block(header: str, description: str):
    print(
        f"## {header}\n\n" +
        f"{description}\n\n" +
        "| Bench | Minimum (s) | Maximum (s) | Average (s) |\n" +
        "| ----- | ----------- | ----------- | ----------- |"
    )


def runtime(name: str, number: int = 1000, repeat: int = 5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Warm-up run
            func(*args, **kwargs)

            def func_to_benchmark():
                func(*args, **kwargs)

            times = timeit.repeat(
                func_to_benchmark,
                number=number,
                repeat=repeat
            )

            min_time = min(times) / number
            max_time = max(times) / number
            avg_time = (sum(times) / len(times)) / number

            print(f"| {name} | {min_time:.3e} | {
                  max_time:.3e} | {avg_time:.3e} |")

            return min_time, avg_time
        return wrapper
    return decorator
