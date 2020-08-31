// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include <iostream>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/wrapper_hand_tracking.pb.h"


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include "mediapipe/framework/port/ret_check.h"

#define PORT     8080
#define MAXLINE 1024
int sockfd;
struct sockaddr_in     servaddr;

using std::vector;

namespace mediapipe {

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormRectTag[] = "NORM_RECTS";
constexpr char kDetectionsTag[] = "DETECTIONS";


void setup_udp(){
  // int sockfd;
  char buffer[MAXLINE];
  // char *hello = "Hello from client";
  // struct sockaddr_in     servaddr;

  // Creating socket file descriptor
  if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
      perror("socket creation failed");
      exit(EXIT_FAILURE);
  }

  memset(&servaddr, 0, sizeof(servaddr));

  // Filling server information
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(PORT);
  servaddr.sin_addr.s_addr = INADDR_ANY;
}


class MyPassThroughCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      cc->Inputs().Get(id).SetAny();
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).Header().IsEmpty()) {
        cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
      }
    }
    //if (cc->OutputSidePackets().NumEntries() != 0) {
    //  for (CollectionItemId id = cc->InputSidePackets().BeginId();
    //       id < cc->InputSidePackets().EndId(); ++id) {
    //    cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
    //  }
    //}
    cc->SetOffset(TimestampDiff(0));


    setup_udp();


    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    //cc->GetCounter("PassThrough")->Increment();

    if (cc->Inputs().NumEntries() == 0) {
      return tool::StatusStop();
    }

    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).IsEmpty()) {


        /*-------------------------------------------------------------------*/
        /*------------ EDITS to original pass_through_calculator ------------*/
        /*-------------------------------------------------------------------*/

        WrapperHandTracking* wrapper = new WrapperHandTracking();
        wrapper->InitAsDefaultInstance();

        if (cc->Inputs().Get(id).Name() == "multi_hand_landmarks"){
          std::cout << "multi_hand_landmarks" << std::endl;

          // the type is a NormalizedLandmarkList, but you need the kLandmarksTag
          // in order for it not to crash for some reason ...
          const vector<NormalizedLandmarkList>& landmarks_vec = cc->Inputs().Tag(kLandmarksTag).Get<vector<NormalizedLandmarkList>>();

          for (NormalizedLandmarkList landmarks : landmarks_vec) {
            for (int i = 0; i < landmarks.landmark_size(); ++i) {
                const NormalizedLandmark& landmark = landmarks.landmark(i);
                std::cout << "Landmark " << i <<":\n" << landmark.DebugString() << '\n';

                wrapper->mutable_landmarks()->add_landmark();
                int size = wrapper->mutable_landmarks()->landmark_size()-1;
                wrapper->mutable_landmarks()->mutable_landmark(size)->set_x(landmark.x());
                wrapper->mutable_landmarks()->mutable_landmark(size)->set_y(landmark.y());
                wrapper->mutable_landmarks()->mutable_landmark(size)->set_z(landmark.z());
            }
          }
        }

        continue; // TODO!!

        if (cc->Inputs().Get(id).Name() == "multi_palm_detections"){
          // Palm is detected once, not continuously — when it first shows up in the image
          const auto& detections = cc->Inputs().Tag(kDetectionsTag).Get<vector<Detection>>();
          for (int i = 0; i < detections.size(); ++i) {
              const Detection& detection = detections[i];
              // std::cout << "\n----- Detection -----\n " << detection.DebugString() << '\n';
              // wrapper->mutable_detection()->add_detection();

          }
        }

        if (cc->Inputs().Get(id).Name() == "multi_hand_rect"){
          // The Hand Rect is an x,y center, width, height, and angle (in radians)
          const NormalizedRect& rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();

          wrapper->mutable_rect()->set_x_center(rect.x_center());
          wrapper->mutable_rect()->set_y_center(rect.y_center());
          wrapper->mutable_rect()->set_width(rect.width());
          wrapper->mutable_rect()->set_height(rect.height());
          wrapper->mutable_rect()->set_rotation(rect.rotation());


          // std::cout << "Hand Rect: " << rect.DebugString() << '\n';
          std::string msg_buffer;
          rect.SerializeToString(&msg_buffer);
          sendto(sockfd, msg_buffer.c_str(), msg_buffer.length(),
              0, (const struct sockaddr *) &servaddr,
                  sizeof(servaddr));

        }

        std::string msg_buffer;
        wrapper->SerializeToString(&msg_buffer);

        sendto(sockfd, msg_buffer.c_str(), msg_buffer.length(),
            0, (const struct sockaddr *) &servaddr,
                sizeof(servaddr));

      /*-------------------------------------------------------------------*/

        VLOG(3) << "Passing " << cc->Inputs().Get(id).Name() << " to "
                << cc->Outputs().Get(id).Name() << " at "
                << cc->InputTimestamp().DebugString();
        cc->Outputs().Get(id).AddPacket(cc->Inputs().Get(id).Value());
      }
    }
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Close(CalculatorContext* cc) {
    if (!cc->GraphStatus().ok()) {
      return ::mediapipe::OkStatus();
    }
    close(sockfd);
    return ::mediapipe::OkStatus();
  }

};

REGISTER_CALCULATOR(MyPassThroughCalculator);

}
