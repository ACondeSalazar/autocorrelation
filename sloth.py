"""
Small library to cache functions results on disk
"""

import numpy as np
import tifffile
import hashlib
import os
import pickle
import functools
from typing import Any, Callable, Optional, Union

CACHE_ROOT = ".sloth_cache"


def myhash(obj: Any, warn: bool = True) -> str:
    """
    Generate a consistent SHA1 hash for any object.
    Handles numpy arrays, tuples, lists, dicts, Dask arrays, and general objects.
    
    Args:
        obj: Object to hash
        warn: If True, emit warnings for potentially unreliable hashes
        
    Returns:
        SHA1 hash string
        
    Warnings:
        Emits UserWarning if hashing may be unreliable (e.g., using repr())
    """
    # Dask arrays and collections - hash by metadata not object identity
    if hasattr(obj, '__dask_graph__'):
        try:
            parts = [
                'dask',
                str(obj.shape) if hasattr(obj, 'shape') else '',
                str(obj.dtype) if hasattr(obj, 'dtype') else '',
                str(obj.chunks) if hasattr(obj, 'chunks') else '',
                str(obj.name) if hasattr(obj, 'name') else '',
            ]
            # Include Dask keys if available (helps identify source)
            if hasattr(obj, '__dask_keys__'):
                try:
                    keys = obj.__dask_keys__()
                    parts.append(str(keys))
                except Exception:
                    pass
            # Include base path if from file
            if hasattr(obj, 'filename'):
                parts.append(str(obj.filename))
            if hasattr(obj, 'url'):
                parts.append(str(obj.url))
            return hashlib.sha1('|'.join(parts).encode()).hexdigest()
        except Exception:
            # Fallthrough to pickle for Dask objects that fail metadata extraction
            pass
    
    if isinstance(obj, np.ndarray):
        return hashlib.sha1(obj.tobytes()).hexdigest()
    elif isinstance(obj, (bytes, bytearray)):
        return hashlib.sha1(obj).hexdigest()
    elif isinstance(obj, str):
        return hashlib.sha1(obj.encode('utf-8')).hexdigest()
    elif isinstance(obj, (tuple, list)):
        # Recursively hash each element
        hasher = hashlib.sha1()
        for item in obj:
            hasher.update(myhash(item, warn=warn).encode('utf-8'))
        return hasher.hexdigest()
    elif isinstance(obj, dict):
        # Sort items for consistent hashing
        sorted_items = sorted(obj.items(), key=lambda x: myhash(x[0], warn=warn))
        hasher = hashlib.sha1()
        for k, v in sorted_items:
            hasher.update(myhash(k, warn=warn).encode('utf-8'))
            hasher.update(myhash(v, warn=warn).encode('utf-8'))
        return hasher.hexdigest()
    else:
        # Try to pickle for general objects
        try:
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.sha1(serialized).hexdigest()
        except (TypeError, pickle.PicklingError):
            # Fallback to string representation - unreliable!
            if warn:
                import warnings
                warnings.warn(
                    f"Object of type {type(obj).__name__} cannot be reliably hashed. "
                    f"Using repr() which may cause cache misses for equivalent objects. "
                    f"Consider converting to a hashable type.",
                    UserWarning,
                    stacklevel=3
                )
            return hashlib.sha1(repr(obj).encode('utf-8')).hexdigest()


def _serialize(obj: Any) -> bytes:
    """Serialize object using optimal method."""
    if isinstance(obj, np.ndarray):
        # Use numpy's efficient binary format
        import io
        buffer = io.BytesIO()
        np.save(buffer, obj, allow_pickle=True)
        return buffer.getvalue()
    else:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(data: bytes) -> Any:
    """Deserialize object using optimal method."""
    try:
        # Try numpy format first
        import io
        buffer = io.BytesIO(data)
        return np.load(buffer, allow_pickle=True)
    except (ValueError, TypeError):
        # Fallback to pickle
        return pickle.loads(data)


def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate a unique cache key for a function call."""
    # Function identifier: module.qualname + bytecode hash
    func_identifier = f"{func.__module__}.{func.__qualname__}"
    func_bytecode_hash = hashlib.sha1(func.__code__.co_code).hexdigest()
    
    # Hash arguments
    args_hash = myhash(args)
    
    # Hash keyword arguments (sorted by key for consistency)
    sorted_kwargs = sorted(kwargs.items(), key=lambda x: x[0])
    kwargs_hash = myhash(sorted_kwargs)
    
    return f"{func_identifier}:{func_bytecode_hash}:{args_hash}:{kwargs_hash}"


class SlothCache:
    """
    Persistent cache for intermediate computation results.
    Stores results on disk and retrieves them across runs.
    """
    
    def __init__(self, path: str, verbose: int = 0):
        """
        Initialize cache with a path (subdirectory under CACHE_ROOT).
        
        Args:
            path: Subdirectory name for this cache
            verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        """
        os.makedirs(CACHE_ROOT, exist_ok=True)
        self.cache_dir = os.path.join(CACHE_ROOT, path)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.verbose = verbose
        
        if verbose >= 1:
            print(f"SlothCache initialized at: {self.cache_dir}")

    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        h = myhash(key)
        return os.path.join(self.cache_dir, f"{h}.cache")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value if it exists."""
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        
        if self.verbose >= 2:
            print(f"Cache hit: {key}")
        
        with open(path, 'rb') as f:
            data = f.read()
        return _deserialize(data)

    def put(self, key: str, obj: Any) -> None:
        """Store a value in the cache."""
        path = self._get_cache_path(key)
        data = _serialize(obj)
        
        with open(path, 'wb') as f:
            f.write(data)
        
        if self.verbose >= 2:
            print(f"Cached: {key}")

    def get_or_compute(self, key: str, compute_func: Callable[[], Any]) -> Any:
        """
        Get value from cache, or compute and cache it if not present.
        
        Args:
            key: Cache key
            compute_func: Function to call if cache miss
            
        Returns:
            Cached or newly computed value
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        
        result = compute_func()
        self.put(key, result)
        return result

    def remove(self, key: str) -> bool:
        """Remove a specific cached item."""
        path = self._get_cache_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def clear(self) -> int:
        """Remove all cached items. Returns count of removed items."""
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                filepath = os.path.join(self.cache_dir, filename)
                os.remove(filepath)
                count += 1
        return count

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        path = self._get_cache_path(key)
        return os.path.exists(path)

    def log(self, key: str, message: str) -> None:
        """Log a message if verbose level allows."""
        if self.verbose >= 2:
            print(f"[{key}] {message}")


# Default cache instance
_default_cache = None


def get_default_cache() -> SlothCache:
    """Get or create the default SlothCache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = SlothCache("default", verbose=1)
    return _default_cache


def sloth_cache(
    path_or_cache: Optional[Union[str, SlothCache]] = None,
    verbose: int = 0
) -> Callable[[Callable], Callable]:
    """
    Decorator to automatically cache function results persistently.
    
    Can be used in several ways:
    
    @sloth_cache()
    def func(...): ...
    
    @sloth_cache("my_cache")
    def func(...): ...
    
    @sloth_cache(my_cache_instance)
    def func(...): ...
    
    Args:
        path_or_cache: Cache path string or SlothCache instance
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Resolve cache instance
        if path_or_cache is None:
            cache = get_default_cache()
        elif isinstance(path_or_cache, SlothCache):
            cache = path_or_cache
        else:
            cache = SlothCache(path_or_cache, verbose=verbose)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = _make_cache_key(func, args, kwargs)
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                if verbose >= 1:
                    print(f"[CACHE HIT] {func.__name__}")
                return cached_result
            
            # Compute and cache
            if verbose >= 1:
                print(f"[CACHE MISS] {func.__name__}")
            
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Add cache management methods to the wrapper
        wrapper.cache = cache
        wrapper.cache_key = lambda *args, **kwargs: _make_cache_key(func, args, kwargs)
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator


# =============================================================================
# DEFAULT EXPORT
# =============================================================================

# Create default cache instance
cache = get_default_cache()

# For backward compatibility, keep the old interface
SlothCache.cache = cache


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sloth Automatic Caching")
    print("=" * 60)
    
    # Test 1: Basic hash function
    print("\n[TEST 1] Hash function")
    img1 = tifffile.imread("./tmp/img1.tif")
    img2 = tifffile.imread("./tmp/img2.tif")
    h1 = myhash(img1)
    h2 = myhash(img2)
    print(f"img1 hash: {h1[:16]}...")
    print(f"img2 hash: {h2[:16]}...")
    print(f"Same image hashes match: {h1 == myhash(img1)}")
    
    # Test 2: Decorator-based caching
    print("\n[TEST 2] Decorator caching")
    
    call_count = 0
    
    @sloth_cache("test_functions")
    def expensive_operation(data):
        global call_count
        call_count += 1
        print(f"  Computing... (call #{call_count})")
        # Simulate expensive computation
        import time
        time.sleep(0.5)
        return data * 2
    
    # First call - should compute
    result1 = expensive_operation("test_input")
    print(f"  Result: {result1}, Calls: {call_count}")
    
    # Second call with same input - should use cache
    result2 = expensive_operation("test_input")
    print(f"  Result: {result2}, Calls: {call_count}")
    
    # Third call with different input - should compute
    result3 = expensive_operation("different_input")
    print(f"  Result: {result3}, Calls: {call_count}")
    
    # Fourth call with first input again - should use cache
    result4 = expensive_operation("test_input")
    print(f"  Result: {result4}, Calls: {call_count}")
    
    # Test 3: Numpy array caching
    print("\n[TEST 3] Caching with numpy arrays")
    
    np_call_count = 0
    
    @sloth_cache("numpy_tests", verbose=1)
    def process_array(arr):
        global np_call_count
        np_call_count += 1
        print(f"  Processing array... (call #{np_call_count})")
        return arr.sum()
    
    arr = np.random.rand(100, 100)
    
    r1 = process_array(arr)
    print(f"  Result: {r1}, Calls: {np_call_count}")
    
    r2 = process_array(arr)
    print(f"  Result: {r2}, Calls: {np_call_count}")
    
    # Test 4: Cache invalidation
    print("\n[TEST 4] Cache invalidation")
    
    @sloth_cache("invalidation_test")
    def counted_func(x):
        global call_count
        call_count += 1
        return x * call_count
    
    # Clear the call count
    call_count = 0
    
    v1 = counted_func("a")
    print(f"  First call: {v1}, Calls: {call_count}")
    
    v2 = counted_func("a")
    print(f"  Cached call: {v2}, Calls: {call_count}")
    
    # Manually clear cache
    counted_func.clear_cache()
    
    v3 = counted_func("a")
    print(f"  After clear: {v3}, Calls: {call_count}")
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)
    
    input("\nPress Enter to exit...")
