//
//  OCRViewController.m
//  OCR
//
//  Created by uhou on 16/1/9.
//  Copyright © 2016年 shop. All rights reserved.
//

#import "OCRViewController.h"
#import "TesseractOCR.h"
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/opencv.hpp>
#import "CVTools.h"
#import "Photo.h"
#import "BundleTools.h"
#define WIDTH_SCREEN [UIScreen mainScreen].bounds.size.width
#define HEIGHT_SCREEN [UIScreen mainScreen].bounds.size.height
using namespace cv;

@interface OCRViewController()<G8TesseractDelegate, CvVideoCameraDelegate>

@property (strong, nonatomic)  UIImageView *targetImageView;
@property (strong, nonatomic)  UIImageView *cameraImageVIew;
@property (strong, nonatomic)  UIView *recognizeTargetView;
@property (strong, nonatomic)  UIButton *photoButton;
@property (strong, nonatomic)  UILabel *textLabel;

@property (strong, nonatomic) NSOperationQueue *operationQueue;
@property (retain, nonatomic) AVCaptureDevice *videoDevice;
@property (retain, nonatomic) CvVideoCamera *videoCamera;
@property (retain, nonatomic) UIImage *currentImage;
@property (strong, nonatomic) dispatch_queue_t cropImageQueue;

@end
@implementation OCRViewController
@synthesize targetImageView,cameraImageVIew,recognizeTargetView,photoButton,textLabel;

-(instancetype)init {
    self = [super init];
    return self;
    
}

- (void)viewDidLoad {
    [super viewDidLoad];
    

    [self setViewUp];
    self.operationQueue = [[NSOperationQueue alloc] init];
   
    self.cropImageQueue = dispatch_queue_create("crop_queue", nil);

   
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for ( AVCaptureDevice *device in devices)
    {
        if ( AVCaptureDevicePositionBack == [device position])
        {
            self.videoDevice = device;
        }
    }
    UITapGestureRecognizer *tapGer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(cameraFocusAction)];
    [self.cameraImageVIew addGestureRecognizer:tapGer];
}

-(void)setViewUp {
    cameraImageVIew = [[UIImageView alloc] initWithFrame:CGRectMake(0, 0, [UIScreen mainScreen].bounds.size.width, [UIScreen mainScreen].bounds.size.height)];
    cameraImageVIew.userInteractionEnabled = YES;
    [self.view addSubview:cameraImageVIew];
    
    textLabel = [[UILabel alloc] initWithFrame:CGRectMake(0, 30, 320, 40)];
    textLabel.textAlignment = NSTextAlignmentCenter;
    textLabel.center = CGPointMake([UIScreen mainScreen].bounds.size.width/2, textLabel.center.y);
    textLabel.backgroundColor = [UIColor clearColor];
    [self.view addSubview:textLabel];
    
    targetImageView = [[UIImageView alloc] initWithFrame:CGRectMake(0, 0, 260, 36)];
    targetImageView.center = CGPointMake(textLabel.center.x, textLabel.center.y + 50);
    [self.view addSubview:targetImageView];
    recognizeTargetView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 260, 36)];
    recognizeTargetView.backgroundColor = [UIColor whiteColor];
    recognizeTargetView.alpha = 0.5;
    recognizeTargetView.center = CGPointMake(WIDTH_SCREEN/2, HEIGHT_SCREEN/2);
    [self.view addSubview:recognizeTargetView];
    
    UIView *bottomBgView = [[UIView alloc] initWithFrame:CGRectMake(0, HEIGHT_SCREEN - 120, WIDTH_SCREEN, 120)];
    bottomBgView.backgroundColor = [UIColor blackColor];
    bottomBgView.alpha = 0.3;
    
    photoButton = [[UIButton alloc] initWithFrame:CGRectMake(0, 0, 84, 84)];
    photoButton.center = CGPointMake(bottomBgView.frame.size.width/2, bottomBgView.frame.size.height/2);
//    NSLog(@"=========================");
//    NSString *image_url = [[[NSBundle mainBundle] resourcePath] stringByAppendingPathComponent:@"Resources.bundle/photo@2x.png"];
    UIImage *image = [UIImage imageWithContentsOfFile: [BundleTools getBundlePath: @"photo@2x.png"]];
    [photoButton setImage:image forState:UIControlStateNormal];
    [photoButton addTarget:self action:@selector(photoAction:) forControlEvents:UIControlEventTouchUpInside];
    [bottomBgView addSubview:photoButton];
    [self.view addSubview:bottomBgView];
    
    
}

- (void)willRotateToInterfaceOrientation:(UIInterfaceOrientation)toInterfaceOrientation duration:(NSTimeInterval)duration {
    [self.videoCamera adjustLayoutToInterfaceOrientation:toInterfaceOrientation];
}
// 当前viewcontroller是否支持转屏
- (BOOL)shouldAutorotate {
    return YES;
}
//当前viewcontroller支持哪些转屏方向
- (NSUInteger)supportedInterfaceOrientations {
    return UIInterfaceOrientationMaskAll;
}
// 当前viewcontroller默认的屏幕方向
-(UIInterfaceOrientation)preferredInterfaceOrientationForPresentation {
    return UIInterfaceOrientationPortrait;
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    
    if (self.videoCamera == nil) {
        self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.cameraImageVIew];
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
        self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetHigh;
        self.videoCamera.defaultFPS = 20;
        self.videoCamera.grayscaleMode = NO;
        self.videoCamera.delegate = self;
//        [self.videoCamera adjustLayoutToInterfaceOrientation:UIInterfaceOrientationLandscapeRight];
        [self.videoCamera start];
    }
    

}

- (void)cameraFocusAction {
    [self cameraFocusAtTarget];
}

- (IBAction)photoAction:(id)sender {
    
    dispatch_async(self.cropImageQueue, ^{

        [self cropByTarget:^(UIImage *image) {
            
            //Start OCR -----
            dispatch_sync(dispatch_get_main_queue(), ^{
//                [self.recognizeTargetView setInnerImage:image];
//                [self ocrStarting];
                
                
                [self doRecognition:image complete:^(NSString *recognizedText) {
//                    [self ocrFinished:recognizedText];
                    //Finish OCR ----
                    NSLog(@"%@",recognizedText);
                    
                    
                    self.textLabel.text = recognizedText;
                }];
                
            });
        }];
    });
    
}

-(void) cameraFocusAtTarget
{
    CGFloat x = self.recognizeTargetView.center.x / self.view.bounds.size.width;
    CGFloat y = self.recognizeTargetView.center.y / self.view.bounds.size.height;
    CGPoint point = CGPointMake(x, y);
    if ([self.videoDevice isFocusPointOfInterestSupported] && [self.videoDevice isFocusModeSupported:AVCaptureFocusModeAutoFocus]) {
        NSError *error;
        if ([self.videoDevice lockForConfiguration:&error]) {
            [self.videoDevice setFocusPointOfInterest:point];
            [self.videoDevice setFocusMode:AVCaptureFocusModeAutoFocus];
            [self.videoDevice unlockForConfiguration];
        } else {
            NSLog(@"Error in Focus Mode");
        }
    }
}

- (void)processImage:(cv::Mat&)image
{
    // OpenCV convert to scanable mode
    //image = [self imageScanableProcessing:image];
    
    
//    if (self.textDectionMode == 1) {
        //textDetection
        /*
        std::vector<cv::Rect> letterBBoxes= [CVTools detectLetters:image];
        for(int i=0; i< letterBBoxes.size(); i++){
//            cv::rectangle(image,letterBBoxes[i],cv::Scalar(0,255,0),3,8,0);
        }
        */
//    }
    
   
    
    self.currentImage = [CVTools UIImageFromCVMat:image];
    
    
//    [self cropByTarget:nil];
}



- (void) cropByTarget:(void(^)(UIImage *image))completion{
    
   
    
    UIImage *image = self.currentImage;
    CGRect wrapperRect = self.cameraImageVIew.frame;
    CGRect frameRect = self.recognizeTargetView.frame;
    
    CGFloat scaleW = image.size.width / wrapperRect.size.width;
    CGFloat scaleH = image.size.height / wrapperRect.size.height;
    
    CGRect rect = CGRectMake(
                             (frameRect.origin.x )*scaleW,
                             (frameRect.origin.y )*scaleH,
                             frameRect.size.width*scaleW,
                             (frameRect.size.height )*scaleH
                             );
    
    CGImageRef imageRef = CGImageCreateWithImageInRect([image CGImage], rect);
    UIImage *cropedImg = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    
    
    cv::Mat mat = [CVTools cvMatFromUIImage:cropedImg];
//    std::vector<cv::Rect> letterBBoxes= [CVTools detectLetters:mat];
//    for(int i=0; i< letterBBoxes.size(); i++) {
//        cv::rectangle(mat,letterBBoxes[i],cv::Scalar(0,255,0),3,8,0);
//    }
    //  同态滤波
//    cv::Mat dest;
//    HomoFilter(mat, dest);
    // 高斯模糊 去噪
    mat = [CVTools gaussianBlurForCvMat:mat];
    cropedImg = [CVTools UIImageFromCVMat:mat];
    
    
    // Apply openCV effect
    cropedImg = [CVTools UIImageFromCVMat:[self imageScanableProcessing:[CVTools cvMatFromUIImage:cropedImg]]];
    cropedImg = [CVTools Erzhiimage:[CVTools UIImageFromCVMat:mat]];
    if (completion!=nil) {
        completion(cropedImg);
    } else {
//        dispatch_async(dispatch_get_main_queue(), ^{
//            self.targetImageView.image = cropedImg;
//        });
    }
    
}
#pragma mark - openCV

- (cv::Mat) imageScanableProcessing:(cv::Mat)image{
    cv::Mat image_copy;
    cvtColor(image, image_copy, CV_BGRA2BGR);
    image = image_copy;
    bitwise_not(image, image_copy);
    image = image_copy;
    
    cvtColor(image, image_copy, CV_RGBA2GRAY);
    image = image_copy;
    return image;
}



#pragma mark - OCR

- (void) doRecognition:(UIImage*)image complete:(void(^)(NSString *recognizedText))complete{
    
    // Mark below for avoiding BSXPCMessage error
    //UIImage *bwImage =[image g8_blackAndWhite];
    
    G8RecognitionOperation *operation = [[G8RecognitionOperation alloc] initWithLanguage:@"eng"];
    operation.tesseract.maximumRecognitionTime = 10.0;
    //operation.tesseract.engineMode = G8OCREngineModeTe sseractOnly;
    //operation.tesseract.pageSegmentationMode = G8PageSegmentationModeSingleLine;
    UIImage  *g8 = [image g8_blackAndWhite];
    operation.delegate = self;
    operation.tesseract.image = g8;
    
    operation.tesseract.charWhitelist = @"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    operation.tesseract.charBlacklist = @".|\\/,';:`~-_^";
    operation.recognitionCompleteBlock = ^(G8Tesseract *tesseract) {
        NSString *recognizedText = tesseract.recognizedText;
        NSLog(@"recognizedText= %@", recognizedText);
        complete(recognizedText);
        self.targetImageView.image = g8;
        [Photo savedPhotosAlbumwithImage:g8];
        [G8Tesseract clearCache];
    };
    
    [self.operationQueue addOperation:operation];
}


- (void)progressImageRecognitionForTesseract:(G8Tesseract *)tesseract {
    
}

- (BOOL)shouldCancelImageRecognitionForTesseract:(G8Tesseract *)tesseract {
    return NO;
}

@end
