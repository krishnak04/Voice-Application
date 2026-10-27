import os
import sys

# Automatically discover and preload pip-installed nvidia cuda libraries on Linux
if sys.platform.startswith("linux"):
    try:
        import ctypes

        # Preload nvJitLink first (required by cuBLAS 12.9+)
        try:
            import nvidia.nvjitlink.lib
            jit_dir = os.path.dirname(nvidia.nvjitlink.lib.__file__)
            for f in sorted(os.listdir(jit_dir)):
                if f.startswith("libnvJitLink") and ".so" in f:
                    try:
                        ctypes.CDLL(os.path.join(jit_dir, f), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
        except Exception:
            pass

        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        cublas_dir = os.path.dirname(nvidia.cublas.lib.__file__)
        cudnn_dir = os.path.dirname(nvidia.cudnn.lib.__file__)

        for lib in ["libcublasLt.so.12", "libcublas.so.12"]:
            p = os.path.join(cublas_dir, lib)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

        for fname in sorted(os.listdir(cudnn_dir)):
            if fname.startswith("libcudnn") and ".so" in fname:
                try:
                    ctypes.CDLL(os.path.join(cudnn_dir, fname), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
    except Exception:
        pass

import uvicorn
import config

if __name__ == "__main__":
    print(f"Starting Voice Benchmark API server on http://{config.HOST}:{config.PORT}")
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False
    )
