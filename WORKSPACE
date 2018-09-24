http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

# When changing the TensorFlow version, also update tools/rf_bazel.rc.
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-7832d2c3a84c79c0dc76a7ed1f6560707a294f22",
    urls = ["https://github.com/tensorflow/tensorflow/archive/7832d2c3a84c79c0dc76a7ed1f6560707a294f22.zip"],
)
#local_repository(
#    name = "org_tensorflow",
#    path = "../tensorflow",
#)

new_http_archive(
    name = "com_google_benchmark",
    build_file = "cc/benchmark.BUILD",
    strip_prefix = "benchmark-1.3.0",
    urls = ["https://github.com/google/benchmark/archive/v1.3.0.zip"],
)

http_archive(
    name = "com_googlesource_code_cctz",
    strip_prefix = "cctz-2.2",
    urls = ["https://github.com/google/cctz/archive/v2.2.zip"],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace()

load("//cc:tensorrt_parsers_configure.bzl", "tensorrt_parsers_configure")

tensorrt_parsers_configure(name = "local_config_tensorrt_parsers")
