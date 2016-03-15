//
//  OCRManager.m
//  OCR
//
//  Created by Dylan on 16/3/4.
//  Copyright © 2016年 Dylan.Lee. All rights reserved.
//

#import "OCRManager.h"

@implementation OCRManager
+(void)showOCRViewWithFrame:(CGRect)frame {
    OCRViewController *ocrVC= [[OCRViewController alloc] init];
    [ocrVC.view setFrame:CGRectMake(0, 0, frame.size.width, frame.size.height)];
    UIView *bgView = [[UIView alloc] initWithFrame:frame];
    bgView.tag = 2999;
    [bgView addSubview:ocrVC.view];
    
    UIWindow *keywindow = [[[UIApplication sharedApplication] delegate] window];
    if (![keywindow.subviews containsObject:bgView])
    {
        // Add overlay
        
        [keywindow addSubview:bgView];
    }
}

+(void)showOCRView {
    [OCRManager showOCRViewWithFrame:CGRectMake(0, 0, [UIScreen mainScreen].bounds.size.width, [UIScreen mainScreen].bounds.size.height)];
};

+(void)hideOCRView {
    UIWindow *keywindow = [[[UIApplication sharedApplication] delegate] window];
    UIView *modal = [keywindow.subviews objectAtIndex:keywindow.subviews.count-1];
    UIView *overlay = [keywindow.subviews objectAtIndex:keywindow.subviews.count-2];
    [overlay removeFromSuperview];
    [modal removeFromSuperview];
}

+ (OCRViewController *)getOCRViewController {
    OCRViewController *ocrVC= [[OCRViewController alloc] init];
    return ocrVC;
}

@end
