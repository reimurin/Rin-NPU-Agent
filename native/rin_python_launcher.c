#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*py_bytes_main_fn)(int, char **);

int main(int argc, char **argv) {
    const char *lib_path = getenv("RIN_PYTHON_LIB");
    if (lib_path == NULL || lib_path[0] == '\0') {
        fprintf(stderr, "RIN_PYTHON_LAUNCHER: RIN_PYTHON_LIB is not set\n");
        return 120;
    }
    void *handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle == NULL) {
        fprintf(stderr, "RIN_PYTHON_LAUNCHER: dlopen failed: %s\n", dlerror());
        return 121;
    }
    dlerror();
    py_bytes_main_fn py_main = (py_bytes_main_fn)dlsym(handle, "Py_BytesMain");
    const char *error = dlerror();
    if (error != NULL || py_main == NULL) {
        fprintf(stderr, "RIN_PYTHON_LAUNCHER: Py_BytesMain missing: %s\n", error ? error : "unknown");
        return 122;
    }
    return py_main(argc, argv);
}
