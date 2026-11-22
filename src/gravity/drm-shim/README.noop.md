# GPU-less noop DRM shim

`libgravity_null_drm_shim.so` (built from `gravity_noop.c`, not installed)
accepts every ioctl without a GPU, so host builds can run the Gravity
Vulkan driver through shader compilation, state emission and command
construction. Submissions are dropped, so rendered results and readbacks
are meaningless.

Reports an Apple M4 (`g16g` / T8132) GPU.
