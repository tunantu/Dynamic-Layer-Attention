_base_ = './faster_rcnn_r50dlal_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        layers=[3, 4, 23, 3], 
        init_cfg=dict(type='Pretrained',
                      checkpoint='./pretrained/r101_dla.pth.tar')))
