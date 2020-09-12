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

#define PORT     8080

int sockfd;
struct sockaddr_in     servaddr;

using std::vector;

namespace mediapipe {

void setup_udp(){
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


class MyPassThroughCalculatorSingle : public CalculatorBase {
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
    cc->GetCounter("PassThrough")->Increment();
    //std::cout << "======================== " <<  cc->GetCounter("PassThrough")->Get() << std::endl;


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

        if (cc->Inputs().Get(id).Name() == "hand_landmarks"){
          const vector<NormalizedLandmarkList>& landmarks{cc->Inputs().Get(id).Get<NormalizedLandmarkList>()};

          for (int j = 0; j < landmarks.size(); ++j) {
            //std::cout << "Landmarks " << j << std::endl;
            wrapper->add_landmarks();

            for (int i = 0; i < landmarks[j].landmark_size(); ++i) {
                const NormalizedLandmark& landmark = landmarks[j].landmark(i);
                //std::cout << "Landmark " << j << " " << i << " \n";// << landmark.DebugString() << '\n';
                wrapper->mutable_landmarks(j)->add_landmark();
                wrapper->mutable_landmarks(j)->mutable_landmark(i)->set_x(landmark.x());
                wrapper->mutable_landmarks(j)->mutable_landmark(i)->set_y(landmark.y());
                wrapper->mutable_landmarks(j)->mutable_landmark(i)->set_z(landmark.z());
            }
          }
        }

        if (cc->Inputs().Get(id).Name() == "palm_detections") {
          // TODO I don't understand detections...
          // Palm is detected once, not continuously — when it first shows up in the image
          const vector<Detection>& detections = cc->Inputs().Get(id).Get<vector<Detection>>();

          for (int i = 0; i < detections.size(); ++i) {
              const Detection& detection = detections[i];
              //std::cout << "Detection  " << i << " " <<  detection.score()[0] << " " << detection.label()[0] << '\n';
              wrapper->mutable_detection()->add_detection();
              wrapper->mutable_detection()->mutable_detection(i)->add_score(detection.score()[0]);
          }
        }

        if (cc->Inputs().Get(id).Name() == "hand_rect") {
          const vector<NormalizedRect>& rects {cc->Inputs().Get(id).Get<NormalizedRect>()};

          for (int i = 0; i < rects.size(); i++) {
            wrapper->add_rects();
            wrapper->mutable_rects(i)->set_x_center(rects[i].x_center());
            wrapper->mutable_rects(i)->set_y_center(rects[i].y_center());
            wrapper->mutable_rects(i)->set_width(rects[i].width());
            wrapper->mutable_rects(i)->set_height(rects[i].height());
            wrapper->mutable_rects(i)->set_rotation(rects[i].rotation());

            //std::cout << "Hand Rect: " << i << '\n';
          }
        }

        std::string msg_buffer;
        wrapper->SerializeToString(&msg_buffer);

        if (msg_buffer.length() > 0) {
          //std::cout << msg_buffer.length() << " " <<  wrapper->landmarks().size() << std::endl;
          sendto(sockfd, msg_buffer.c_str(), msg_buffer.length(),
              0, (const struct sockaddr *) &servaddr,
                  sizeof(servaddr));
        }

      /*-------------------------------------------------------------------*/

        //VLOG(3) << "Passing " << cc->Inputs().Get(id).Name() << " to "
        //        << cc->Outputs().Get(id).Name() << " at "
        //        << cc->InputTimestamp().DebugString();
        //cc->Outputs().Get(id).AddPacket(cc->Inputs().Get(id).Value());
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

REGISTER_CALCULATOR(MyPassThroughCalculatorSingle);

}

