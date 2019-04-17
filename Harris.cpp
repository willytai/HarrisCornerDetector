#include "Harris.h"
#include <iostream>

void HarrisFeatureDetector::read_image(char* filename) {
    std::cerr << std::endl << "[ Reading image \"" << filename << "\" ]" << std::endl;
    _image_rgb = cv::imread(filename, 1);
    if (!_image_rgb.data) {
        std::cerr << "\t> Failed! Could not open or find the image!" << std::endl;
    }
    else {
        std::cerr << "\t> Success! Image scale " << _image_rgb.rows << " x " << _image_rgb.cols << std::endl;
        std::cerr << "\t> Converting to gray scale..." << std::endl;
        cv::cvtColor(_image_rgb, _image_gray, cv::COLOR_RGB2GRAY);
    }
}

void HarrisFeatureDetector::run() {
    this->compute_derivatives();
    this->compute_products();
    this->compute_products_of_sums();
    this->compute_response();
    this->compute_local_max_R();
}

void HarrisFeatureDetector::show_images() const {
    cv::namedWindow("dx", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("dy", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("color", cv::WINDOW_AUTOSIZE);
    cv::imshow("dx", _image_dx);
    cv::imshow("dy", _image_dy);
    cv::imshow("original", _image_gray);
    cv::imshow("color", _image_rgb);
    cv::waitKey(0);
}

void HarrisFeatureDetector::compute_derivatives() {
    std::cerr << std::endl << "[ Computing x and y derivatives ]" << std::endl;
    int guassian_kernel_size = this->create_guassian_kernel_5();
    std::cerr << "\t> bluring image with window = " << guassian_kernel_size << std::endl;
    this->convolution(_image_gray, _image_gray, _guassian_filter, guassian_kernel_size);
    std::cerr << "\t> computing dx" << std::endl;
    cv::Sobel(_image_gray, _image_dx, CV_8U, 2, 0, 5);
    std::cerr << "\t> computing dy" << std::endl;
    cv::Sobel(_image_gray, _image_dy, CV_8U, 0, 2, 5);
    delete _guassian_filter;
}

void HarrisFeatureDetector::compute_products() {
    std::cerr << std::endl << "[ Computing products of derivatives ]" << std::endl;
    std::cerr << "\t> computing IxIx" << std::endl;
    this->multiply(_image_dxdx, _image_dx, _image_dx);
    std::cerr << "\t> computing IyIy" << std::endl;
    this->multiply(_image_dydy, _image_dy, _image_dy);
    std::cerr << "\t> computing IxIy" << std::endl;
    this->multiply(_image_dxdy, _image_dx, _image_dy);
}

void HarrisFeatureDetector::compute_products_of_sums() {
    std::cerr << std::endl << "[ Computing products of sums ]" << std::endl;
    int guassian_kernel_size = this->create_guassian_kernel_3();
    std::cerr << "\t> using guassian filter with window = " << guassian_kernel_size << std::endl;
    std::cerr << "\t> computing Sxx" << std::endl;
    this->convolution(_image_dxdx, _image_Sxx, _guassian_filter, guassian_kernel_size);
    std::cerr << "\t> computing Syy" << std::endl;
    this->convolution(_image_dydy, _image_Syy, _guassian_filter, guassian_kernel_size);
    std::cerr << "\t> computing Sxy" << std::endl;
    this->convolution(_image_dxdy, _image_Sxy, _guassian_filter, guassian_kernel_size);
    delete _guassian_filter;
}

void HarrisFeatureDetector::compute_response() {
    std::cerr << std::endl << "[ Computing response ]" << std::endl;
    _R.resize(_image_Sxx.size());
    for (int row = 0; row < _image_Sxx.size(); ++row) {
        _R[row].resize(_image_Sxx[row].size());
        for (int col = 0; col < _image_Sxx[row].size(); ++col) {
            // detM = a00a11 - a01a10
            // traceM = a00 + a11
            double detM   = _image_Sxx[row][col] * _image_Syy[row][col] - _image_Sxy[row][col] * _image_Sxy[row][col];
            double traceM = _image_Sxx[row][col] + _image_Syy[row][col];
            _R[row][col]  = detM - _k*traceM*traceM;
        }
    }
}

void HarrisFeatureDetector::compute_local_max_R() {
    std::cerr << std::endl << "[ Searching for local max in response ]" << std::endl;
    // local maximun detector's window set to 5
    int offset      = 5;
    int window_size = 5;
    int half_width  = window_size >> 1;
    for (int row = half_width+offset; row < _image_rgb.rows-half_width-offset; ++row) {
        for (int col = half_width+offset; col < _image_rgb.cols-half_width-offset; ++col) {
            if (this->isMax(_R, row, col, half_width)) {
                PRObject curObj(row, col, _R[row][col]);
                if (_result.size() < _features) {
                    _result.insert(curObj);
                    continue;
                }
                else {
                    PRObject min = _result.top();
                    if (min > curObj) continue;
                    _result.pop();
                    _result.insert(curObj);
                }
            }
        }
    }
    std::cerr << "\t> " << _features << " feature points chosen" << std::endl;
}

void HarrisFeatureDetector::dump(std::ostream& os) {
    std::cerr << std::endl << "[ Dumping result ]" << std::endl;
    os << "row,col" << std::endl;
    for (auto it = _result.begin(); it != _result.end(); ++it) {
        int row, col;
        (*it).get_coordinates(row, col);
        os << row << ',' << col << std::endl;
    }
}

void HarrisFeatureDetector::show_result() {
    for (auto it = _result.begin(); it != _result.end(); ++it) {
        int row, col;
        (*it).get_coordinates(row, col);
        uchar* pixel_ptr = _image_rgb.ptr(row, col);
        pixel_ptr[0] = 0;
        pixel_ptr[1] = 0;
        pixel_ptr[2] = 255;
    }
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", _image_rgb);
    cv::waitKey(0);
}
