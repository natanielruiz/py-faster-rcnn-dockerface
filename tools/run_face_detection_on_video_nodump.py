from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys

# Amherst network
# NETS = {'vgg16': ('VGG16',
#           'output/faster_rcnn_end2end/train/vgg16_faster_rcnn_iter_80000.caffemodel')}

# Dockerface network
NETS = {'vgg16': ('VGG16',
          'output/faster_rcnn_end2end/wider/vgg16_dockerface_iter_80000.caffemodel')}

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
            help='Use CPU mode (overrides --gpu)',
            action='store_true')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
            choices=NETS.keys(), default='vgg16')
  parser.add_argument('--video', dest='video_path', help='Path of video')
  parser.add_argument('--output_string', dest='output_string', help='String appended to output file')

  args = parser.parse_args()

  return args

if __name__ == '__main__':
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  # cfg.TEST.BBOX_REG = False

  args = parse_args()

  prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
              'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
  caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                NETS[args.demo_net][1])

  prototxt = 'models/face/VGG16/faster_rcnn_end2end/test.prototxt'
  caffemodel = NETS[args.demo_net][1]

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
             'fetch_faster_rcnn_models.sh?').format(caffemodel))

  if args.cpu_mode:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)

  print '\n\nLoaded network {:s}'.format(caffemodel)

  # LOAD DATA FROM VIDEO
  data_dir = 'data/video/'
  out_dir = 'output/video/'

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  dets_file_name = os.path.join(out_dir, 'video-det-fold-%s.txt' % args.output_string)
  fid = open(dets_file_name, 'w')

  CONF_THRESH = 0.85
  NMS_THRESH = 0.15

  print args.video_path
  if not os.path.exists(args.video_path):
    print 'Video does not exist.'

  video = cv2.VideoCapture(args.video_path)

  # Get width, height
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

  n_frame = 1
  # TODO: add time function per frame.
  while (True):
      ret, frame = video.read()

      if ret == True:
          # frame is BGR cv2 image.
          # # Detect all object classes and regress object bounds
          scores, boxes = im_detect(net, frame)

          cls_ind = 1
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
          cls_scores = scores[:, cls_ind]
          dets = np.hstack((cls_boxes,
                    cls_scores[:, np.newaxis])).astype(np.float32)
          keep = nms(dets, NMS_THRESH)
          dets = dets[keep, :]

          keep = np.where(dets[:, 4] > CONF_THRESH)
          dets = dets[keep]

          # dets are the upper left and lower right coordinates of bbox
          # dets[:, 0] = x_ul, dets[:, 1] = y_ul
          # dets[:, 2] = x_lr, dets[:, 3] = y_lr

          dets[:, 2] = dets[:, 2]
          dets[:, 3] = dets[:, 3]
          if (dets.shape[0] != 0):
              for j in xrange(dets.shape[0]):
                fid.write(str(n_frame) + ' %f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], dets[j, 2], dets[j, 3], dets[j, 4]))
              # Draw bbox
              cv2.rectangle(frame,(int(dets[j, 0]), int(dets[j, 1])),(int(dets[j, 2]), int(dets[j, 3])),(0,255,0),3)
          print 'Detected faces in frame number: ' + str(n_frame)
          n_frame += 1
      else:
          break

  print 'Done with detection.'
  fid.close()
  video.release()
