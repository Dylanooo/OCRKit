//
//  Photo.m
//  OCR
//
//  Created by Dylan on 16/3/4.
//  Copyright © 2016年 Dylan.Lee. All rights reserved.
//

#import "Photo.h"

@implementation Photo

+ (void)savedPhotosAlbumwithImage:(UIImage *) image
{
    UIImageWriteToSavedPhotosAlbum(image, self, @selector(image:didFinishSavingWithError:contextInfo:),NULL);
}

/**
 *  指定回调方法
 *
 *  @param image       图片
 *  @param error       错误
 *  @param contextInfo 上下文
 */
+ (void)image: (UIImage *) image didFinishSavingWithError: (NSError *) error contextInfo: (void *) contextInfo
{
    NSString *msg = nil ;

    if(error != NULL){
        msg = @"保存图片失败" ;
    }else{
        msg = @"保存图片成功" ;
    }
    NSLog(@"%@",msg);
}
@end
