#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
from shutil import copyfile

symlink = False  # If this is false the files are copied instead
combine_train_valid = False  # If this is true, the train and valid sets are ALSO combined
cinic_directory = "data/CINIC-10"
imagenet_directory = "data/CINIC-10-ImageNet"
cifar_directory = "data/CINIC-10-CIFAR"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
sets = ['train', 'valid', 'test']
directory = cifar_directory  # or cifar_directory
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(directory + '/train'):
    os.makedirs(directory + '/train')
if not os.path.exists(directory + '/test'):
    os.makedirs(directory + '/test')

for c in classes:
    if not os.path.exists('{}/train/{}'.format(directory, c)):
        os.makedirs('{}/train/{}'.format(directory, c))
    if not os.path.exists('{}/test/{}'.format(directory, c)):
        os.makedirs('{}/test/{}'.format(directory, c))
    if not combine_train_valid:
        if not os.path.exists('{}/valid/{}'.format(directory, c)):
            os.makedirs('{}/valid/{}'.format(directory, c))

for s in sets:
    for c in classes:
        source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
        filenames = glob.glob('{}/*.png'.format(source_directory))
        for fn in filenames:
            dest_fn = fn.split('/')[-1].split('\\')[-1]
            if (s == 'train' or s == 'valid') and combine_train_valid and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train/{}/{}'.format(directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'train') and 'cifar'  in fn.split('/')[-1]:
                dest_fn = '{}/train/{}/{}'.format(directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'valid') and 'cifar'  in fn.split('/')[-1]:
                dest_fn = '{}/valid/{}/{}'.format(directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif s == 'test' and 'cifar'  in fn.split('/')[-1]:
                dest_fn = '{}/test/{}/{}'.format(directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)