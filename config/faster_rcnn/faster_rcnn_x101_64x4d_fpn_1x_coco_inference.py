_base_ = './faster_rcnn_x101_64x4d_fpn_1x_coco.py'
data_root = 'data/'

data = dict(
    test=dict(
        samples_per_gpu=4,
        ann_file=data_root + 'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json',
        img_prefix=data_root + 'mva2023_sod4bird_pub_test/images/',
    ) 
)

