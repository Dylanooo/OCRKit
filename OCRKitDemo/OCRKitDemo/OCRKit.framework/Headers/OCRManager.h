//
//  OCRManager.h
//  OCR
//
//  Created by Dylan on 16/3/4.
//  Copyright © 2016年 Dylan.Lee. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import "OCRViewController.h"
@interface OCRManager : NSObject
/**
 *  显示ocr 主视图
 *
 *  @param frame 要现实视图的大小
 */
+(void)showOCRViewWithFrame:(CGRect)frame;
/**
 *  显示ocr视图
 */
+(void)showOCRView;
/**
 *  隐藏ocr视图
 */
+(void)hideOCRView;

+ (OCRViewController *)getOCRViewController;
@end
