//
//  Photo.h
//  OCR
//
//  Created by Dylan on 16/3/4.
//  Copyright © 2016年 Dylan.Lee. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
@interface Photo : NSObject
/**
 *  存储图片到相册
 *
 *  @param image 要存储的图片
 */
+ (void)savedPhotosAlbumwithImage:(UIImage *) image;
@end
