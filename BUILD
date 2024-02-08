py_binary(
    name = "mps",
    srcs = [
        "mps/mps.py"
    ],
)

py_library(
    name = "test_lib",
    srcs = glob([
        "mps/tests/*.py",
        "mps/tests/unit_tests/*.py",
        "mps/tests/integration_tests/*.py"
    ]),
    deps = [
        "mps",
    ],
)

py_binary(
    name = "run_tests",
    main = "mps/tests/run_tests.py",
    srcs = [
        "mps/tests/run_tests.py"
    ],
    deps = [
        "test_lib"
    ],
)
