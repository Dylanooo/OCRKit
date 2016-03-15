//
//  BundleTools.m
//  OCRKit
//
//  Created by Dylan on 16/3/4.
//  Copyright © 2016年 Dylan.Lee. All rights reserved.
//

#import "BundleTools.h"
#define BUNDLE_NAME @"OCRResources"
@implementation BundleTools
+ (NSBundle *)getBundle{
    
    return [NSBundle bundleWithPath: [[NSBundle mainBundle] pathForResource: BUNDLE_NAME ofType: @"bundle"]];
}

+ (NSString *)getBundlePath: (NSString *) assetName{
    
    NSBundle *myBundle = [BundleTools getBundle];
    
    if (myBundle && assetName) {
        
        return [[myBundle resourcePath] stringByAppendingPathComponent: assetName];
    }
    
    return nil;
}
@end
