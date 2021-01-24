# build go2digital
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --cxxopt=-std=c++17 go2digital:upper_body_pose_tracking_gpu
