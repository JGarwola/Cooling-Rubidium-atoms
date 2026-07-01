#!/usr/bin/env python3
"""
Test script to check if numba and joblib libraries are working.
"""

def test_numba():
    """Test numba JIT compilation."""
    try:
        from numba import jit
        import numpy as np
        print("numba imported successfully.")
        
        @jit(nopython=True)
        def sum2d(arr):
            s = 0
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    s += arr[i, j]
            return s
        
        # Create a small array and test the function
        arr = np.random.rand(10, 10)
        result = sum2d(arr)
        expected = np.sum(arr)
        print(f"numba test: sum2d result = {result:.6f}, expected = {expected:.6f}")
        if np.allclose(result, expected):
            print("numba test passed.")
            return True
        else:
            print("numba test failed: results do not match.")
            return False
    except ImportError as e:
        print(f"Failed to import numba: {e}")
        return False
    except Exception as e:
        print(f"numba test failed: {e}")
        return False

def test_joblib():
    """Test joblib parallel processing."""
    try:
        from joblib import Parallel, delayed
        import multiprocessing
        print("joblib imported successfully.")
        
        n_cpus = multiprocessing.cpu_count()
        print(f"Number of CPUs available: {n_cpus}")
        
        # Simple function to parallelize
        def square(x):
            return x * x
        
        # Test with n_jobs=-1 (use all CPUs)
        data = list(range(100))
        try:
            result = Parallel(n_jobs=-1)(delayed(square)(i) for i in data)
            expected = [i*i for i in data]
            if result == expected:
                print(f"joblib test passed with n_jobs=-1 (using {n_cpus} CPUs).")
                return True
            else:
                print("joblib test failed: results do not match.")
                return False
        except Exception as e:
            print(f"joblib test failed with n_jobs=-1: {e}")
            # Try with a specific number of jobs, say 2
            try:
                result = Parallel(n_jobs=2)(delayed(square)(i) for i in data)
                expected = [i*i for i in data]
                if result == expected:
                    print(f"joblib test passed with n_jobs=2.")
                    return True
                else:
                    print("joblib test failed: results do not match.")
                    return False
            except Exception as e2:
                print(f"joblib test also failed with n_jobs=2: {e2}")
                return False
    except ImportError as e:
        print(f"Failed to import joblib: {e}")
        return False
    except Exception as e:
        print(f"joblib test failed: {e}")
        return False

def main():
    print("Testing numba and joblib...")
    numba_ok = test_numba()
    joblib_ok = test_joblib()
    
    if numba_ok and joblib_ok:
        print("\nAll tests passed.")
    else:
        print("\nSome tests failed.")
        if not numba_ok:
            print(" - numba test failed.")
        if not joblib_ok:
            print(" - joblib test failed.")

if __name__ == "__main__":
    main()