"""
@Author: yangqiang
@Email: whuhit09@gmail.com
@time: 2020/12/28 5:38 下午
"""
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random

COCO_LABEL_MAP = {1: 1}

COCO_CLASSES = ('tongue',)


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.label_map = COCO_LABEL_MAP

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None,
                 target_transform=COCOAnnotationTransform(),
                 dataset_name='MS COCO'):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())  # 标签数目 小于样本数目，说明有的图像没有标签

        self.transform = transform
        self.target_transform = target_transform

        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        # # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target = self.coco.imgToAnns[img_id]  # 这一句跟下面两句是一样的

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd  # 列表相加
        assert len(target) > 0, f"{img_id} has not labels"

        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, _ = img.shape

        # Pool all the masks for this image into one [num_objects,height,width] matrix
        masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
        masks = np.vstack(masks)
        masks = masks.reshape(-1, height, width)

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                       {'num_crowds': num_crowds, 'labels': target[:, 4]})
            # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
            num_crowds = labels['num_crowds']
            labels = labels['labels']
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        # # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}

        res = {img_id: []}
        for obj in target:
            class_name = COCO_CLASSES[COCO_LABEL_MAP[obj['category_id']] - 1]
            coords = obj["bbox"]
            res[img_id].append([class_name, coords])

        return res


if __name__ == "__main__":
    coco = COCODetection(image_path="/Users/yang/02_data/02_007_Object_Detection/tongueCOCO/mini_jpgs",
                         info_file="/Users/yang/02_data/02_007_Object_Detection/tongueCOCO/mini_val_tongue_coco.json")
    img, target, masks, height, width, num_crowds = coco.pull_item(0)

    print()
    print(img.shape)
    print("target:")
    for tar in target:
        print(tar)
    print("masks", masks.shape)
    print("height, width, num_crowds", height, width, num_crowds)
