//
//  BundleTools.h
//  OCRKit
//
//  Created by Dylan on 16/3/4.
//  Copyright © 2016年 Dylan.Lee. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface BundleTools : NSObject
+ (NSString *)getBundlePath: (NSString *) assetName;
+ (NSBundle *)getBundle;

@end
