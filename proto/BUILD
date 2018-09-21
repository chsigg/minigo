package(default_visibility = [
    ":__pkg__",
    "//cc:__pkg__",
    "//cc/dual_net:__pkg__",
])

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")

cc_proto_library(
    name = "inference_service_proto",
    srcs = ["inference_service.proto"],
    use_grpc_plugin = True,
)

py_proto_library(
    name = "inference_service_py_pb2",
    srcs = ["inference_service.proto"],
    use_grpc_plugin = True,
)
