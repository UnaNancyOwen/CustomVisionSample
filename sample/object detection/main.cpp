#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "util.h"

float logistic( float x )
{
    if( x > 0.0 ){
        return 1.0 / ( 1.0 + std::exp( -x ) );
    }
    else{
        float e = std::exp( x );
        return e / ( 1.0 + e );
    }
}

int main( int argc, char* argv[] )
{
    // Open Video Capture
    cv::VideoCapture capture = cv::VideoCapture( 0 );
    if( !capture.isOpened() ){
        return -1;
    }

    // Read Class Name List and Color Table
    const std::string list = "../labels.txt";
    const std::vector<std::string> classes = readClassNameList( list );
    const std::vector<cv::Scalar> colors = getClassColors( classes.size() );

    // Read Custom Vision Model
    const std::string model = "../model.onnx";
    cv::dnn::Net net = cv::dnn::readNet( model );
    if( net.empty() ){
        return -1;
    }

    /*
        Please see list of supported combinations backend/target.
        https://docs.opencv.org/4.2.0/db/d30/classcv_1_1dnn_1_1Net.html#a9dddbefbc7f3defbe3eeb5dc3d3483f4
    */

    // Set Preferable Backend
    net.setPreferableBackend( cv::dnn::DNN_BACKEND_OPENCV );

    // Set Preferable Target
    net.setPreferableTarget( cv::dnn::DNN_TARGET_OPENCL );

    while( true ){
        // Read Frame
        cv::Mat frame;
        capture >> frame;
        if( frame.empty() ){
            cv::waitKey( 0 );
            break;
        }
        if( frame.channels() == 4 ){
            cv::cvtColor( frame, frame, cv::COLOR_BGRA2BGR );
        }

        // Create Blob from Input Image
        // Custom Vision ( Scale : 1.f, Size : 416 x 416, Mean Subtraction : ( 0.0, 0.0, 0.0 ), Channels Order : BGR )
        cv::Mat resize_frame;
        cv::resize( frame, resize_frame, cv::Size( 416, 416 ) );
        cv::Mat blob = cv::dnn::blobFromImage( resize_frame, 1.f, cv::Size( 416, 416 ), cv::Scalar(), false, false );

        // Set Input Blob
        net.setInput( blob );

        // Run Forward Network
        std::vector<cv::Mat> detections;
        net.forward( detections, getOutputsNames( net ) );

        // Get Object Bounding-Boxes, Confidences, and Class-Indices
        std::vector<int32_t> class_ids; std::vector<float> confidences; std::vector<cv::Rect> rectangles;
        const std::vector<float> anchors = { 0.573f, 0.677f, 1.87f, 2.06f, 3.34f, 5.47f, 7.88f, 3.53f, 9.77f, 9.17f };
        for( cv::Mat& detection : detections ){
            const int32_t num_anchor = anchors.size() / 2;
            const int32_t channels   = detection.size[1];
            const int32_t height     = detection.size[2];
            const int32_t width      = detection.size[3];
            const int32_t num_class  = ( channels / num_anchor ) - 5;

            for( int32_t grid_y = 0; grid_y < height; grid_y++ ){
                for( int32_t grid_x = 0; grid_x < width; grid_x++ ){
                    int32_t offset = 0;
                    const int32_t stride = height * width;
                    const int32_t base_offset = grid_x + grid_y * width;

                    for( int32_t i = 0; i < num_anchor; i++ ){
                        float x = ( logistic( detection.at<float>( base_offset + ( offset++ * stride ) ) ) + grid_x ) / width;
                        float y = ( logistic( detection.at<float>( base_offset + ( offset++ * stride ) ) ) + grid_y ) / height;
                        const float w = std::exp( detection.at<float>( base_offset + ( offset++ * stride ) ) ) * anchors[i * 2] / width;
                        const float h = std::exp( detection.at<float>( base_offset + ( offset++ * stride ) ) ) * anchors[i * 2 + 1] / height;

                        x = x - ( w / 2.0 );
                        y = y - ( h / 2.0 );

                        const float objectness = logistic( detection.at<float>( base_offset + ( offset++ * stride ) ) );

                        std::vector<float> class_probabilities( num_class );
                        for( int32_t j = 0; j < num_class; j++ ){
                            class_probabilities[j] = detection.at<float>( base_offset + ( offset++ * stride ) );
                        }
                        const float max = *std::max_element( class_probabilities.begin(), class_probabilities.end() );
                        for( int32_t j = 0; j < num_class; j++ ){
                            class_probabilities[j] = std::exp( class_probabilities[j] - max );
                        }
                        const float sum = std::accumulate( class_probabilities.begin(), class_probabilities.end(), 0.0 );
                        for( int32_t j = 0; j < num_class; j++ )
                        {
                            class_probabilities[j] *= objectness / sum;
                        }

                        const cv::Rect rectangle = cv::Rect( static_cast<int32_t>( x * frame.cols ), static_cast<int32_t>( y * frame.rows ), static_cast<int32_t>( w * frame.cols ), static_cast<int32_t>( h * frame.rows ) );
                        const std::vector<float>::iterator confidence = std::max_element( class_probabilities.begin(), class_probabilities.end() );
                        const int32_t class_id = std::distance( class_probabilities.begin(), confidence );

                        constexpr float threshold = 0.5;
                        if( threshold > *confidence ){
                            continue;
                        }

                        class_ids.push_back( class_id );
                        confidences.push_back( *confidence );
                        rectangles.push_back( rectangle );
                    }
                }
            }
        }

        // Remove Overlap Bounding-Boxes using Non-Maximum Suppression
        constexpr float confidence_threshold = 0.5; // Confidence
        constexpr float nms_threshold = 0.5; // IoU (Intersection over Union)
        std::vector<int32_t> indices;
        cv::dnn::NMSBoxes( rectangles, confidences, confidence_threshold, nms_threshold, indices );

        // Draw Bounding-Boxes
        for( const int32_t& index : indices ){
            const cv::Rect rectangle = rectangles[index];
            const cv::Scalar color = colors[class_ids[index]];
            constexpr int32_t thickness = 3;
            cv::rectangle( frame, rectangle, color, thickness );
        }

        // Show Image
        cv::imshow( "Object Detection", frame );
        const int32_t key = cv::waitKey( 1 );
        if( key == 'q' ){
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
