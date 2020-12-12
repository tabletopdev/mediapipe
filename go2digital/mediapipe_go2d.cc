#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "pose_landmarks_smoothed";

const std::size_t INIT_BUFFER_SIZE = 1024;



DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

::mediapipe::Status RunMPPGraph() {
  std::cout << "Started mediapipe" << std::endl;

  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  //LOG(INFO) << "Get calculator graph config contents: "
  //          << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Start running the calculator graph.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";

  try {
      // on some systems you may need to reopen stdin in binary mode
      std::freopen(nullptr, "rb", stdin);
      if(std::ferror(stdin)) throw std::runtime_error(std::strerror(errno));

      std::size_t len;
      std::array<uint8, INIT_BUFFER_SIZE> buf;
      std::vector<uint8> input;

      while (true) {
        uint32_t frame_num;
        uint32_t length;
        uint32_t stride;
        uint32_t width;
        std::fread(&frame_num, sizeof(frame_num), 1, stdin);
        std::fread(&length, sizeof(length), 1, stdin);
        std::fread(&stride, sizeof(stride), 1, stdin);
        std::fread(&width, sizeof(width), 1, stdin);
        uint32_t height = length / stride;

        std::cout << frame_num << " " << length << " " << stride << " " << width << " " << height << std::endl;

        while((len = std::fread(buf.data(), sizeof(buf[0]), std::min(length - input.size(), buf.size()), stdin)) > 0) {
            if(std::ferror(stdin) && !std::feof(stdin)) throw std::runtime_error(std::strerror(errno));
            input.insert(input.end(), buf.data(), buf.data() + len);
        }

        if (input.size() == 0) break;

        //cv::Mat camera_frame;

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGBA, width, height,
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);

        input_frame.get()->CopyPixelData(
            mediapipe::ImageFormat::SRGBA,
            stride / 4, // width with padding
            height,
            //width, // width step
            input.data(),
            mediapipe::ImageFrame::kGlDefaultAlignmentBoundary
        );

        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        //camera_frame.copyTo(input_frame_mat);

        // Prepare and add graph input packet.
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        MP_RETURN_IF_ERROR(
            gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph, &gpu_helper, &frame_num]() -> ::mediapipe::Status {
              // Convert ImageFrame to GpuBuffer.
              auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
              auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
              glFlush();
              texture.Release();
              // Send GPU image packet into the graph.
              MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
                  kInputStream, mediapipe::Adopt(gpu_frame.release())
                                    .At(mediapipe::Timestamp(frame_timestamp_us))));
              return ::mediapipe::OkStatus();
            }));

        mediapipe::Packet packet;
        if (!poller.Next(&packet)) break;
        std::cout << "PKT SIZE " << packet.GetProtoMessageLite().ByteSize() << std::endl;

        size_t frame_timestamp_us_after =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

        std::cout << frame_timestamp_us_after - frame_timestamp_us << std::endl;

        input.clear();
      }
  } catch(std::exception const& e) {
      std::cerr << e.what() << '\n';
      exit(EXIT_FAILURE);
  }

  LOG(INFO) << "Shutting down.";
  // if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}