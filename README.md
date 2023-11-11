# TagTracker2

Dependencies can be installed using:

```
pip3 install -r requirements.txt
pip3 install --only-binary :all: --find-links https://tortall.net/~robotpy/wheels/2023/raspbian/ pyntcore robotpy_wpimath robotpy_wpinet
```

WPILib dependencies must be installed from binary, building the wheels locally will cause `MemoryError: std::bad_alloc`.
