from time import perf_counter
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        final_func_name = func.__name__
        print(f'Function {final_func_name} started')
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'Function {final_func_name} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
