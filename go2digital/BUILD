# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "face_effect",
    deps = [
        ":mediapipe_go2d",

        "//mediapipe/graphs/face_effect:face_effect_gpu_deps",
        "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",

        "//mediapipe/graphs/pose_tracking:upper_body_pose_tracking_gpu_deps",
        "//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_gpu_deps"
    ],
    data = [
        "//mediapipe/graphs/face_effect/data:glasses",
        "//mediapipe/graphs/face_effect:face_effect_gpu.binarypb",
        "//mediapipe/graphs/face_effect/data:facepaint.pngblob",
        "//mediapipe/graphs/face_effect/data:glasses.binarypb",
        "//mediapipe/graphs/face_effect/data:glasses.pngblob",
        "//mediapipe/modules/face_detection:face_detection_front.tflite",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
    ],
)

cc_library(
    name = "mediapipe_go2d",
    srcs = ["mediapipe_go2d.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
    ],
)

cc_library(
    name = "mediapipe_go2d_old",
    srcs = ["demo_run_graph_main_gpu.cc"],
    data = [
        "//mediapipe/graphs/face_effect/data:glasses",
        "//mediapipe/graphs/face_effect:face_effect_gpu.binarypb",
        "//mediapipe/graphs/face_effect/data:facepaint.pngblob",
        "//mediapipe/graphs/face_effect/data:glasses.binarypb",
        "//mediapipe/graphs/face_effect/data:glasses.pngblob",
        "//mediapipe/modules/face_detection:face_detection_front.tflite",
        "//mediapipe/modules/face_geometry/data:geometry_pipeline_metadata.binarypb",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
    ],
)

