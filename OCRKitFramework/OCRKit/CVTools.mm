//
//  CVTools.m
//  ios-snapSearch-live
//
//  Created by Carter Chang on 7/19/15.
//  Copyright (c) 2015 Carter Chang. All rights reserved.
//

#import "CVTools.h"
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/opencv.hpp>
#import <opencv2/core/core.hpp>
#import <opencv2/imgproc/imgproc.hpp>
#define D0_GHPF 80
@implementation CVTools

+(std::vector<cv::Rect>) detectLetters:(cv::Mat)img{
    std::vector<cv::Rect> boundRect;
    cv::Mat img_gray, img_sobel, img_threshold, element;
    cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    IplImage grey = img_gray;
    cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    unsigned char* dataImage = (unsigned char*)grey.imageData;
    int threshold = OTSU(dataImage, grey.width, grey.height);
    cv::threshold(img_sobel, img_threshold, threshold, 255, CV_THRESH_BINARY);
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(23, 2) );
    cv::morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
    std::vector< std::vector< cv::Point> > contours;
    cv::findContours(img_threshold, contours, 0, 1);
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    for( int i = 0; i < contours.size(); i++ ){
        if (contours[i].size()>600)
        {
            cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
            cv::Rect appRect( boundingRect( cv::Mat(contours_poly[i]) ));
            if (appRect.width>appRect.height)
                boundRect.push_back(appRect);
        }
    }
    return boundRect;
}


// Ref: http://docs.opencv.org/doc/tutorials/ios/image_manipulation/image_manipulation.html
+(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

// Ref: http://docs.opencv.org/doc/tutorials/ios/image_manipulation/image_manipulation.html
+ (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

+ (cv:: Mat)gaussianBlurForCvMat:(cv::Mat)cvMat {
    cv::Mat outPutImg;

    for (int i = 0;i < 10; i ++) {
        GaussianBlur(cvMat,outPutImg,cvSize(5,5),0,0);
    }
    
    return outPutImg;
}

void HomoFilter(Mat srcImg, Mat &dst)
{
    srcImg.convertTo(srcImg, CV_64FC1);
    dst = Mat::zeros(srcImg.rows, srcImg.cols, CV_64FC1);
    
    // 构造滤波矩阵
    Mat H_u_v;
    double gammaH = 1.5;
    double gammaL = 0.5;
    double C = 1;
    double d0 = (srcImg.rows/2)*(srcImg.rows/2) + (srcImg.cols/2)*(srcImg.cols/2);
    double d2 = 0;
    H_u_v = Mat::zeros(srcImg.rows, srcImg.cols, CV_64FC1);
    for (int i = 0; i < srcImg.rows; i++)
    {
        double * dataH_u_v = H_u_v.ptr<double>(i);
        for (int j = 0; j < srcImg.cols; j++)
        {
            d2 = pow((i - srcImg.rows/2), 2.0) + pow((j - srcImg.cols/2), 2.0);
            dataH_u_v[j] =  (gammaH - gammaL)*(1 - exp(-C*d2/d0)) + gammaL;
        }
    }
    
    for (int i = 0; i < srcImg.rows; i++)
    {
        double* srcdata = srcImg.ptr<double>(i);
        double* logdata = dst.ptr<double>(i);
        for (int j = 0; j < srcImg.cols; j++)
        {
            logdata[j] = log(srcdata[j]+1.0);
        }
    }
    
    //%%%%%%%%%%%%%%%%%%%%%%%傅里叶变换、滤波、傅里叶反变换%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Mat planes[] = {dst, Mat::zeros(dst.size(), CV_64F)};
    Mat complexI;
    merge(planes, 2, complexI); // Add to the expanded another plane with zeros
    dft(complexI, complexI);    // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);
    Mat IDFT[] = {Mat::zeros(dst.size(), CV_64F), Mat::zeros(dst.size(), CV_64F)};
    IDFT[0] = H_u_v.mul(planes[0]);//planes[0].mul(H_u_v);
    IDFT[1] = H_u_v.mul(planes[1]);//planes[1].mul(H_u_v);
    
    merge(IDFT, 2, complexI);
    idft(complexI, complexI);
    split(complexI, IDFT);
    
    for (int i = 0; i < srcImg.rows; i++)
    {
        double* dataRe = IDFT[0].ptr<double>(i);
        double* dataIm = IDFT[1].ptr<double>(i);
        double* logdata = dst.ptr<double>(i);
        
        for (int j = 0; j < srcImg.cols; j++)
        {
            if (dataIm[j] < 0)
            {
                logdata[j]  = dataRe[j]*dataRe[j] - dataIm[j]*dataIm[j];
            }
            else
            {
                logdata[j]  = dataRe[j]*dataRe[j] + dataIm[j]*dataIm[j];
            }
        }
    }
    
    
    normalize(dst, dst, 0, 5.545, CV_MINMAX);
    
    for (int i = 0; i < srcImg.rows; i++)
    {
        double* logdata = dst.ptr<double>(i);
        for (int j = 0; j < srcImg.cols; j++)
        {
            logdata[j] = pow(2.718281828, logdata[j]) - 1.0;
        }
    }
    dst.convertTo(dst, CV_8UC1);;
    
}

// OSTU算法求出阈值
int  OTSU(unsigned char* pGrayImg , int iWidth , int iHeight)
{
    if((pGrayImg==0)||(iWidth<=0)||(iHeight<=0))return -1;
    int ihist[256];
    int thresholdValue=0; // „–÷µ
    int n, n1, n2 ;
    double m1, m2, sum, csum, fmax, sb;
    int i,j,k;
    memset(ihist, 0, sizeof(ihist));
    n=iHeight*iWidth;
    sum = csum = 0.0;
    fmax = -1.0;
    n1 = 0;
    for(i=0; i < iHeight; i++)
    {
        for(j=0; j < iWidth; j++)
        {
            ihist[*pGrayImg]++;
            pGrayImg++;
        }
    }
    pGrayImg -= n;
    for (k=0; k <= 255; k++)
    {
        sum += (double) k * (double) ihist[k];
    }
    for (k=0; k <=255; k++)
    {
        n1 += ihist[k];
        if(n1==0)continue;
        n2 = n - n1;
        if(n2==0)break;
        csum += (double)k *ihist[k];
        m1 = csum/n1;
        m2 = (sum-csum)/n2;
        sb = (double) n1 *(double) n2 *(m1 - m2) * (m1 - m2);
        if (sb > fmax)
        {
            fmax = sb;
            thresholdValue = k;
        }
    }
    return(thresholdValue);
}


+(UIImage *)Erzhiimage:(UIImage *)srcimage{
    
    UIImage *resimage;
    cv::Mat matImage = [CVTools cvMatFromUIImage:srcimage];
//    cv::Mat finalBinary;
//    std::vector<cv::Point> cP;
//    cP.push_back(cv::Point(0, 0));
//    cP.push_back(cv::Point(64, 192));
//    cP.push_back(cv::Point(192, 64));
//    cP.push_back(cv::Point(255, 255));
//    adjustCurve(matImage, finalBinary, cP);
//    matImage = finalBinary;
    cv::Mat matGrey;
    
    //5.cvtColor函数对matImage进行灰度处理
    //取得IplImage形式的灰度图像
    cv::cvtColor(matImage, matGrey, CV_BGR2GRAY);// 转换成灰色
    
    //6.使用灰度后的IplImage形式图像，用OSTU算法算阈值：threshold
    IplImage grey = matGrey;
    unsigned char* dataImage = (unsigned char*)grey.imageData;
    int threshold = OTSU(dataImage, grey.width, grey.height);
    printf("阈值：%d\n",threshold);
    
    //7.利用阈值算得新的cvMat形式的图像
    cv::Mat matBinary;
    cv::threshold(matGrey, matBinary, threshold, 255, cv::THRESH_BINARY);
    
    //8.cvMat形式的图像转UIImage
    cv::Mat resultBinary;
    sharpenImage1(matBinary, resultBinary);
    UIImage* image = [[UIImage alloc ]init];
    image = [CVTools UIImageFromCVMat:resultBinary];
    
    resimage = image;
    
    return resimage;
}

void adjustHist(Mat& src, Mat& dst)
{
    std::vector<Mat> bgr(3);
    std::vector<Mat> bgr2(3);
    split(src, bgr);
    equalizeHist(bgr[0], bgr2[0]);
    equalizeHist(bgr[1], bgr2[1]);
    equalizeHist(bgr[2], bgr2[2]);
    merge(bgr2, dst);
}


void adjustCurve(Mat& src, Mat& dst, std::vector<cv::Point>& cP)
{
    // calculate curve
    int numP = cP.size();
    Mat A(numP, numP, CV_64F);
    Mat    B(numP, 1, CV_64F);
    Mat X(numP, 1, CV_64F);
    double* pA;
    double* pB;
    double* pX;
    for (int i = 0; i < numP; i++)
    {
        pA = A.ptr<double>(i);
        pB = B.ptr<double>(i);
        cv::Point p = cP[i];
        for (int j = 0; j < numP; j++)
            pA[j] = pow(p.x, j);
        pB[0] = p.y;
    }
    solve(A, B, X, DECOMP_SVD);
    // plot curve
    Mat pCurve = Mat::zeros(256, 256, CV_8U);
    std::vector<int> checkTable(256);
    for (int i = 0; i < 256; i++)
    {
        double y = 0;
        for (int j = 0; j < numP; j++)
        {
            pX = X.ptr<double>(j);
            y += pow(i, j)*pX[0];
        }
        cv::Point p(i, 256 - y);
        if (p.y < 0)
            p.y = 0;
        if (p.y > 255)
            p.y = 255;
        circle(pCurve, p, 1, 255);
        checkTable[i] = 256 - p.y;
    }
//    imshow("Curve", pCurve);
    // adjust the frame
    dst = src.clone();
    int c = src.channels();
    uchar* pSrc;
    uchar* pDst;
    for (int i = 0; i < src.rows; i++)
    {
        pSrc = src.ptr<uchar>(i);
        pDst = dst.ptr<uchar>(i);
        for (int j = 0; j < c*src.cols; j++)
        {
            int x = pSrc[j];
            pDst[j] = checkTable[x];
        }
    }
}




void sharpenImage0(const cv::Mat &image, cv::Mat &result)
 {
         //为输出图像分配内存
         result.create(image.size(),image.type());
    
         /*滤波核为拉普拉斯核3x3：
            25                              0 -1 0
            26                             -1 5 -1
            27                              0 -1 0
            28     */
         for(int j= 1; j<image.rows-1; ++j)
             {
                     const uchar *previous = image.ptr<const uchar>(j-1);
                     const uchar *current = image.ptr<const uchar>(j);
                     const uchar *next = image.ptr<const uchar>(j+1);
                     uchar *output = result.ptr<uchar>(j);
                     for(int i= 1; i<image.cols-1; ++i)
                         {
                                 *output++ = cv::saturate_cast<uchar>(5*current[i]-previous[i]-next[i]-current[i-1]-current[i+1]);  //saturate_cast<uchar>()保证结果在uchar范围内
                             }
                 }
         result.row(0).setTo(cv::Scalar(0));
         result.row(result.rows-1).setTo(cv::Scalar(0));
         result.col(0).setTo(cv::Scalar(0));
         result.col(result.cols-1).setTo(cv::Scalar(0));
     }

 void sharpenImage1(const cv::Mat &image, cv::Mat &result)
{
        //创建并初始化滤波模板
         cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
         kernel.at<float>(1,1) = 5.0;
         kernel.at<float>(0,1) = -1.0;
         kernel.at<float>(1,0) = -1.0;
         kernel.at<float>(1,2) = -1.0;
         kernel.at<float>(2,1) = -1.0;

         result.create(image.size(),image.type());
    
         //对图像进行滤波
         cv::filter2D(image,result,image.depth(),kernel);
}

// int main(int argc, char* argv[])
// {
//         cv::Mat image = cv::imread("../boldt.jpg");
//         cv::Mat image_gray;
//         image_gray.create(image.size(),image.type());
//    
//         if(!image.data)
//                 return -1;
//         if(image.channels() == 3)
//                 cv::cvtColor(image,image_gray,CV_RGB2GRAY);
//    
//         cv::Mat result;
//         result.create(image_gray.size(),image_gray.type());
//        double time_ = static_cast<double>(cv::getTickCount());
//        sharpenImage0(image_gray,result);
//         time_ = 1000*(static_cast<double>(cv::getTickCount())-time_)/cv::getTickFrequency();
//         std::cout<<"time = "<<time_<<"ms"<<std::endl;
//    
//         cv::namedWindow("Image 1");
//         cv::imshow("Image 1",result);
//    
//         cv::Mat result1;
//         result1.create(image_gray.size(),image_gray.type());
//         time_ = static_cast<double>(cv::getTickCount());
//         sharpenImage1(image_gray,result1);
//         time_ = 1000*static_cast<double>(cv::getTickCount()-time_)/cv::getTickFrequency();
//         std::cout<<"time = "<<time_<<"ms"<<std::endl;
//    
//         cv::namedWindow("Image 2");
//         cv::imshow("Image 2",result1);
//    
//         cv::waitKey();
//    return 0;
// }

//
//IplImage* makeupForLight(cv::Mat src) {
//     IplImage pSrcImage = src;
//     IplImage *pImageChannel[4] = {0,0,0,0};
//     IplImage pImage = *cvCreateImage(cvGetSize(&pSrcImage), pSrcImage.depth, pSrcImage.nChannels);
//
//        for(int i=0; i<pSrcImage.nChannels; i++)
//        {
//            pImageChannel[i] = cvCreateImage( cvGetSize(&pSrcImage), pSrcImage.depth, 1);
//        }
//        // 信道分离
//        cvSplit( &pSrcImage, pImageChannel[0], pImageChannel[1],pImageChannel[2],NULL);
//        for(int i = 0; i < pImage.nChannels; i++ )
//        {
//            //直方图均衡化
//            cvEqualizeHist(pImageChannel[i], pImageChannel[i]);
//        }
//        // 信道组合
//        cvMerge( pImageChannel[0], pImageChannel[1], pImageChannel[2],NULL,&pImage);
//        // ……图像显示代码（略）
//        // 释放资源
//        for(int i=0; i<pSrcImage.nChannels; i++)
//        {
//            if(pImageChannel[i])
//            {
//                cvReleaseImage( &pImageChannel[i] );
//                pImageChannel[i] = 0;
//            }
//        };
//     return &pImage;
//}

@end
