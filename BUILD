py_binary(
    name = "mps",
    srcs = [
        "rmbs/mps.py"
    ],
)

py_library(
    name = "test_lib",
    srcs = glob([
        "rmbs/tests/*.py",
        "rmbs/tests/unit_tests/*.py",
        "rmbs/tests/integration_tests/*.py"
    ]),
    deps = [
        "mps",
    ],
)

py_binary(
    name = "run_tests",
    main = "rmbs/tests/run_tests.py",
    srcs = [
        "rmbs/tests/run_tests.py"
    ],
    deps = [
        "test_lib"
    ],
)