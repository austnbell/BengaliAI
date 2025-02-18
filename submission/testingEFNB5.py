# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:54:19 2020

@author: Austin Bell
"""

import numpy as np  # noqa
import pandas as pd
import argparse
import tensorflow as tf
from tqdm.auto import tqdm

from tensorflow.keras import layers as L
import efficientnet.tfkeras as efn

root_path="./data/raw/"

def normalize(image):
  # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/main.py#L325-L326
  # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L31-L32
  image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
  image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
  return image


def get_model(input_size, backbone='efficientnet-b5', weights='noisy-student', tta=False):
  print(f'Using backbone {backbone} and weights {weights}')
  x = L.Input(shape=input_size, name='imgs', dtype='float32')
  y = normalize(x)
  if backbone.startswith('efficientnet'):
    model_fn = getattr(efn, f'EfficientNetB{backbone[-1]}')

  y = model_fn(input_shape=input_size, weights=weights, include_top=False)(y)
  y = L.GlobalAveragePooling2D()(y)
  y = L.Dropout(0.2)(y)
  # 1292 of 1295 are present
  y = L.Dense(1292, activation='softmax')(y)
  model = tf.keras.Model(x, y)

  if tta:
    assert False, 'This does not make sense yet'
    x_flip = tf.reverse(x, [2])  # 'NHWC'
    y_tta = tf.add(model(x), model(x_flip)) / 2.0
    tta_model = tf.keras.Model(x, y_tta)
    return model, tta_model

  return model


import cv2
import numpy as np
import os


def normalize_image(img, org_width, org_height, new_width, new_height):
  # Invert
  img = 255 - img
  # Normalize
  img = (img * (255.0 / img.max())).astype(np.uint8)
  # Reshape
  img = img.reshape(org_height, org_width)
  image_resized = cv2.resize(img, (new_width, new_height))
  return image_resized



def decode_predictions_v2(y_pred, inv_tuple_map):
  # return predictions as tuple (root / 168, vowel / 11, consonant / 7) & ti 1292
  rr = np.zeros((len(y_pred), 168), dtype=np.float32)
  vv = np.zeros((len(y_pred), 11), dtype=np.float32)
  cc = np.zeros((len(y_pred), 7), dtype=np.float32)

  for ti in range(y_pred.shape[1]):
    r_index, v_index, c_index = inv_tuple_map[ti]
    y_pred_ti = y_pred[:, ti]
    rr[:, r_index] += y_pred_ti
    vv[:, v_index] += y_pred_ti
    cc[:, c_index] += y_pred_ti

  decoded = []
  for k in range(len(y_pred)):
    decoded.append(rr[k,:])
    decoded.append(vv[k,:])
    decoded.append(cc[k,:])
    
  return decoded


def process_batch(image_id_batch, img_batch, row_id, target, model, inv_tuple_map):
  img_batch = np.float32(img_batch)
  # deal with single image
  if img_batch.ndim != 4:
    img_batch = np.expand_dims(img_batch, 0)
  y_pred = model.predict(img_batch)
  decoded = decode_predictions_v2(y_pred, inv_tuple_map)
  for iid in image_id_batch:
    row_id.append(iid + '_grapheme_root')
    row_id.append(iid + '_vowel_diacritic')
    row_id.append(iid + '_consonant_diacritic')
  return decoded


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=123)
  parser.add_argument('--input_size', type=str, default='160,256')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--backbone', type=str, default='efficientnet-b5')
  parser.add_argument('--weights', type=str, default='./savedModels/model-B5-25-TPU.h5')
  args, _ = parser.parse_known_args()

  org_height = 137
  org_width = 236
  tuple_map = {(15, 9, 5): 64, (72, 4, 0): 522, (18, 0, 0): 76, (62, 7, 0): 418, (18, 7, 4): 84, (136, 0, 2): 1064, (43, 0, 0): 261, (149, 7, 5): 1184, (165, 7, 0): 1279, (86, 1, 0): 653, (17, 2, 0): 73, (115, 2, 0): 892, (139, 3, 0): 1088, (142, 3, 0): 1117, (23, 10, 0): 149, (150, 1, 0): 1189, (13, 0, 4): 21, (96, 7, 5): 744, (130, 0, 0): 1016, (133, 2, 0): 1035, (60, 1, 0): 407, (81, 4, 4): 615, (64, 4, 0): 437, (107, 0, 0): 801, (115, 9, 2): 907, (125, 1, 0): 993, (25, 2, 0): 155, (153, 2, 5): 1219, (153, 1, 0): 1215, (149, 9, 0): 1185, (103, 1, 1): 776, (148, 9, 0): 1170, (37, 2, 0): 215, (139, 0, 2): 1081, (74, 2, 0): 546, (86, 0, 5): 652, (83, 9, 5): 636, (155, 3, 0): 1232, (133, 4, 4): 1043, (70, 0, 0): 480, (135, 7, 0): 1061, (38, 1, 0): 219, (28, 9, 0): 166, (150, 4, 0): 1195, (81, 5, 4): 617, (59, 2, 0): 397, (108, 2, 0): 835, (69, 3, 0): 478, (140, 7, 0): 1100, (129, 1, 0): 1011, (31, 1, 0): 189, (118, 0, 0): 922, (96, 4, 1): 737, (98, 7, 0): 757, (32, 3, 0): 196, (43, 0, 2): 262, (123, 1, 0): 966, (157, 9, 0): 1241, (72, 6, 2): 529, (89, 2, 5): 675, (8, 0, 0): 12, (94, 0, 0): 709, (113, 0, 4): 857, (72, 0, 4): 507, (163, 1, 0): 1272, (151, 0, 0): 1202, (139, 7, 2): 1092, (79, 8, 0): 598, (144, 7, 0): 1127, (160, 0, 0): 1264, (13, 2, 0): 29, (29, 1, 4): 174, (22, 7, 0): 111, (72, 4, 4): 524, (57, 0, 0): 379, (111, 1, 0): 846, (149, 1, 5): 1176, (127, 4, 0): 1001, (162, 1, 0): 1270, (42, 9, 1): 260, (83, 2, 0): 631, (66, 2, 0): 465, (147, 2, 1): 1142, (97, 1, 0): 752, (53, 1, 0): 327, (120, 0, 0): 939, (167, 7, 0): 1290, (81, 10, 0): 625, (70, 1, 0): 482, (91, 2, 0): 693, (23, 2, 2): 127, (81, 5, 0): 616, (110, 0, 0): 842, (13, 9, 0): 49, (50, 2, 0): 312, (113, 4, 4): 869, (140, 2, 5): 1097, (147, 1, 0): 1136, (64, 6, 2): 443, (66, 0, 0): 463, (123, 0, 0): 965, (159, 2, 2): 1252, (79, 2, 4): 588, (88, 3, 0): 667, (53, 0, 0): 323, (124, 1, 0): 981, (143, 7, 0): 1123, (86, 7, 5): 661, (29, 1, 1): 172, (134, 2, 0): 1057, (70, 0, 4): 481, (133, 1, 0): 1031, (71, 2, 5): 495, (156, 2, 0): 1237, (107, 1, 6): 811, (29, 9, 0): 184, (147, 8, 0): 1155, (117, 6, 0): 919, (115, 7, 0): 902, (64, 0, 5): 423, (71, 1, 2): 491, (147, 4, 4): 1147, (123, 5, 0): 972, (44, 1, 0): 285, (89, 0, 0): 669, (94, 9, 0): 714, (133, 3, 0): 1039, (81, 3, 0): 612, (128, 9, 0): 1009, (16, 9, 0): 70, (23, 0, 4): 118, (21, 0, 2): 92, (98, 0, 0): 754, (15, 0, 0): 58, (72, 3, 2): 520, (76, 8, 0): 569, (96, 2, 1): 731, (132, 9, 0): 1026, (146, 7, 0): 1131, (18, 7, 0): 83, (59, 0, 0): 391, (61, 0, 0): 409, (51, 2, 0): 316, (92, 3, 0): 702, (133, 3, 5): 1041, (165, 2, 0): 1276, (81, 0, 2): 605, (115, 5, 0): 899, (112, 1, 4): 851, (89, 7, 0): 679, (147, 6, 0): 1150, (113, 4, 0): 867, (103, 7, 0): 788, (107, 1, 5): 810, (43, 4, 2): 275, (107, 2, 2): 813, (64, 7, 1): 445, (10, 0, 0): 16, (42, 1, 4): 253, (72, 1, 2): 512, (115, 3, 2): 895, (107, 8, 0): 830, (76, 9, 0): 570, (129, 7, 0): 1014, (95, 7, 0): 719, (122, 9, 0): 963, (141, 0, 5): 1103, (92, 1, 0): 699, (107, 4, 2): 819, (20, 3, 0): 90, (113, 5, 4): 873, (72, 4, 2): 523, (86, 9, 0): 662, (74, 0, 0): 542, (122, 7, 0): 960, (120, 7, 0): 945, (96, 0, 5): 724, (147, 1, 1): 1137, (137, 0, 0): 1070, (151, 2, 0): 1206, (29, 2, 0): 176, (151, 7, 4): 1209, (165, 4, 0): 1278, (103, 1, 0): 775, (90, 7, 0): 687, (123, 1, 4): 968, (147, 0, 4): 1134, (167, 9, 0): 1291, (36, 9, 0): 213, (153, 2, 0): 1218, (67, 0, 0): 468, (23, 8, 0): 144, (25, 0, 0): 152, (157, 1, 0): 1240, (54, 2, 0): 349, (56, 0, 5): 364, (159, 5, 0): 1257, (86, 1, 5): 654, (107, 7, 1): 826, (43, 0, 5): 265, (81, 1, 4): 609, (123, 9, 0): 975, (64, 2, 2): 431, (79, 7, 2): 596, (142, 7, 0): 1119, (93, 0, 0): 705, (147, 1, 5): 1140, (133, 7, 5): 1050, (156, 6, 0): 1239, (32, 0, 0): 193, (23, 2, 0): 126, (132, 1, 0): 1021, (51, 0, 0): 315, (159, 1, 0): 1247, (62, 3, 0): 416, (133, 0, 2): 1028, (136, 7, 0): 1068, (148, 0, 0): 1160, (22, 0, 0): 97, (71, 0, 4): 489, (132, 3, 0): 1023, (150, 9, 5): 1201, (43, 8, 4): 281, (101, 9, 0): 770, (36, 7, 0): 212, (96, 4, 5): 738, (79, 4, 0): 590, (72, 0, 2): 506, (31, 0, 0): 188, (3, 0, 1): 5, (68, 3, 0): 473, (118, 3, 0): 925, (55, 9, 1): 361, (150, 3, 0): 1193, (43, 9, 4): 283, (103, 9, 0): 790, (93, 7, 0): 708, (137, 7, 0): 1073, (98, 2, 0): 756, (13, 5, 5): 41, (113, 2, 0): 864, (50, 1, 0): 311, (113, 5, 0): 871, (125, 7, 0): 996, (55, 0, 0): 351, (43, 2, 2): 270, (138, 0, 0): 1074, (71, 9, 0): 503, (96, 0, 2): 722, (23, 7, 2): 141, (154, 9, 0): 1228, (153, 1, 4): 1216, (29, 4, 0): 177, (9, 0, 0): 13, (91, 4, 0): 695, (64, 2, 0): 430, (40, 7, 0): 245, (28, 1, 0): 162, (38, 4, 4): 227, (141, 7, 4): 1111, (42, 1, 2): 252, (74, 3, 0): 547, (64, 1, 0): 425, (142, 0, 0): 1113, (167, 5, 0): 1289, (91, 7, 4): 697, (107, 7, 0): 825, (152, 1, 0): 1210, (75, 7, 2): 558, (85, 3, 0): 644, (143, 0, 0): 1120, (53, 1, 2): 328, (115, 5, 2): 900, (96, 1, 6): 729, (23, 9, 1): 146, (150, 7, 0): 1198, (124, 7, 0): 988, (96, 9, 5): 748, (81, 2, 0): 610, (107, 4, 5): 821, (76, 2, 0): 563, (29, 1, 0): 171, (96, 2, 0): 730, (117, 2, 0): 915, (147, 9, 4): 1157, (140, 6, 0): 1099, (113, 4, 2): 868, (139, 2, 2): 1087, (117, 4, 0): 917, (165, 1, 0): 1275, (72, 9, 5): 538, (79, 0, 4): 581, (30, 7, 0): 187, (147, 10, 0): 1159, (31, 4, 0): 191, (138, 2, 0): 1075, (13, 9, 2): 50, (83, 2, 5): 632, (59, 7, 4): 404, (115, 4, 4): 898, (22, 0, 4): 99, (71, 0, 0): 486, (107, 7, 4): 828, (121, 1, 0): 948, (113, 7, 4): 878, (133, 7, 4): 1049, (79, 4, 5): 592, (55, 1, 4): 355, (150, 7, 5): 1199, (165, 3, 0): 1277, (55, 3, 0): 357, (88, 0, 0): 664, (109, 0, 0): 836, (35, 0, 0): 204, (147, 0, 2): 1133, (60, 0, 0): 406, (96, 9, 0): 746, (48, 1, 1): 302, (71, 7, 4): 501, (56, 2, 2): 370, (110, 2, 0): 843, (13, 4, 1): 36, (150, 0, 0): 1187, (103, 1, 4): 777, (35, 1, 0): 205, (24, 0, 0): 150, (27, 0, 0): 160, (145, 9, 0): 1129, (42, 7, 0): 258, (79, 2, 0): 586, (68, 7, 0): 474, (115, 1, 0): 888, (133, 2, 4): 1037, (156, 3, 0): 1238, (60, 4, 0): 408, (25, 1, 4): 154, (53, 2, 0): 332, (22, 8, 0): 113, (29, 1, 5): 175, (133, 5, 0): 1045, (56, 7, 5): 376, (85, 2, 0): 643, (64, 7, 2): 446, (132, 4, 0): 1024, (88, 2, 0): 666, (113, 9, 0): 880, (85, 0, 0): 640, (147, 3, 2): 1145, (115, 0, 0): 884, (22, 9, 1): 115, (67, 1, 0): 469, (161, 0, 0): 1267, (59, 0, 4): 393, (139, 7, 4): 1093, (48, 0, 2): 300, (13, 4, 2): 37, (22, 6, 0): 110, (69, 2, 0): 477, (119, 1, 5): 932, (17, 9, 0): 75, (126, 4, 0): 997, (115, 7, 4): 904, (128, 7, 0): 1008, (72, 1, 0): 510, (72, 5, 5): 527, (89, 1, 5): 673, (64, 0, 2): 421, (149, 9, 5): 1186, (133, 3, 2): 1040, (118, 4, 0): 926, (142, 0, 4): 1114, (131, 2, 0): 1018, (128, 1, 0): 1006, (18, 3, 0): 81, (71, 2, 0): 493, (64, 0, 6): 424, (13, 7, 1): 44, (149, 3, 0): 1180, (72, 2, 2): 516, (79, 10, 0): 600, (13, 1, 1): 24, (103, 4, 5): 787, (147, 1, 4): 1139, (135, 9, 0): 1062, (13, 10, 0): 52, (71, 8, 0): 502, (81, 8, 0): 622, (166, 0, 0): 1281, (159, 0, 2): 1244, (52, 1, 0): 318, (72, 0, 0): 505, (81, 9, 0): 623, (79, 5, 0): 593, (124, 3, 0): 986, (150, 2, 0): 1191, (133, 7, 0): 1047, (80, 0, 2): 602, (140, 1, 0): 1096, (39, 1, 0): 238, (109, 7, 0): 841, (48, 9, 0): 307, (38, 7, 4): 233, (17, 7, 0): 74, (27, 2, 0): 161, (151, 7, 0): 1208, (58, 10, 0): 390, (113, 4, 5): 870, (92, 7, 0): 704, (127, 9, 0): 1003, (117, 1, 0): 914, (111, 0, 2): 845, (154, 5, 0): 1227, (106, 0, 0): 796, (123, 3, 0): 970, (64, 0, 4): 422, (48, 4, 0): 304, (74, 4, 0): 548, (23, 9, 0): 145, (71, 7, 0): 499, (13, 4, 5): 39, (72, 2, 0): 515, (115, 9, 0): 906, (72, 9, 0): 535, (22, 4, 1): 109, (86, 3, 5): 658, (139, 9, 0): 1094, (140, 0, 0): 1095, (76, 7, 0): 568, (53, 4, 5): 339, (24, 1, 0): 151, (145, 0, 0): 1128, (143, 1, 0): 1121, (132, 2, 0): 1022, (56, 1, 4): 367, (91, 3, 0): 694, (38, 7, 0): 230, (64, 4, 4): 439, (116, 1, 0): 910, (65, 0, 4): 456, (128, 0, 0): 1004, (122, 7, 2): 961, (75, 7, 0): 557, (38, 1, 2): 221, (113, 5, 2): 872, (23, 1, 6): 125, (133, 1, 4): 1033, (127, 2, 0): 1000, (72, 2, 4): 517, (136, 2, 0): 1066, (47, 2, 0): 298, (103, 9, 5): 792, (25, 1, 0): 153, (119, 3, 0): 934, (106, 1, 0): 797, (79, 1, 2): 584, (151, 5, 0): 1207, (119, 7, 0): 937, (65, 3, 0): 459, (75, 1, 0): 554, (122, 0, 2): 951, (67, 7, 0): 470, (94, 7, 0): 713, (89, 4, 0): 678, (43, 1, 0): 266, (72, 0, 5): 508, (159, 0, 4): 1245, (154, 3, 0): 1225, (161, 1, 0): 1268, (123, 1, 1): 967, (64, 8, 5): 450, (84, 7, 0): 639, (107, 1, 0): 806, (58, 0, 4): 384, (112, 7, 0): 854, (81, 2, 2): 611, (114, 1, 0): 883, (32, 4, 0): 197, (96, 0, 1): 721, (149, 4, 0): 1182, (53, 7, 4): 343, (120, 2, 0): 942, (55, 0, 4): 352, (152, 7, 0): 1212, (81, 7, 2): 620, (46, 1, 0): 293, (96, 6, 0): 740, (79, 1, 0): 582, (160, 2, 0): 1265, (159, 2, 0): 1251, (149, 0, 0): 1171, (107, 2, 4): 814, (152, 2, 0): 1211, (56, 0, 2): 363, (112, 0, 0): 849, (13, 7, 0): 43, (91, 1, 4): 692, (141, 1, 4): 1105, (133, 9, 0): 1052, (53, 1, 5): 330, (102, 0, 0): 771, (56, 3, 0): 372, (96, 3, 5): 735, (89, 0, 5): 671, (65, 1, 0): 457, (148, 1, 4): 1163, (88, 7, 0): 668, (147, 9, 0): 1156, (148, 1, 0): 1162, (59, 1, 2): 395, (118, 2, 0): 924, (153, 1, 5): 1217, (98, 1, 0): 755, (13, 0, 0): 19, (79, 3, 0): 589, (80, 7, 2): 603, (103, 4, 1): 786, (137, 2, 0): 1071, (54, 0, 0): 347, (119, 9, 0): 938, (162, 3, 0): 1271, (13, 0, 2): 20, (59, 7, 2): 403, (92, 2, 0): 701, (113, 10, 0): 882, (107, 2, 0): 812, (107, 4, 4): 820, (139, 0, 0): 1080, (76, 4, 0): 566, (38, 8, 0): 234, (147, 4, 0): 1146, (23, 7, 0): 139, (71, 3, 2): 497, (29, 5, 0): 179, (90, 3, 0): 686, (66, 3, 0): 466, (159, 6, 0): 1258, (113, 5, 5): 874, (117, 7, 0): 920, (128, 2, 0): 1007, (42, 0, 2): 249, (159, 8, 0): 1262, (123, 10, 0): 977, (94, 4, 0): 712, (55, 2, 0): 356, (85, 1, 0): 642, (64, 7, 0): 444, (43, 9, 0): 282, (72, 8, 0): 534, (23, 1, 1): 121, (38, 9, 0): 235, (103, 3, 0): 783, (150, 3, 5): 1194, (144, 0, 0): 1124, (1, 0, 0): 1, (72, 6, 0): 528, (61, 2, 0): 411, (96, 7, 2): 743, (42, 2, 0): 254, (83, 1, 5): 630, (64, 0, 0): 420, (71, 3, 0): 496, (44, 9, 0): 290, (147, 1, 2): 1138, (150, 5, 0): 1196, (116, 3, 0): 911, (13, 1, 4): 26, (25, 3, 0): 156, (153, 0, 0): 1214, (81, 0, 0): 604, (111, 4, 0): 848, (23, 2, 4): 128, (165, 0, 0): 1274, (89, 7, 5): 681, (32, 6, 0): 198, (122, 4, 0): 958, (107, 0, 2): 802, (48, 2, 0): 303, (23, 7, 5): 143, (96, 2, 5): 733, (79, 2, 2): 587, (81, 1, 2): 608, (22, 1, 1): 101, (22, 1, 2): 102, (91, 0, 0): 688, (86, 7, 0): 660, (42, 2, 1): 255, (96, 1, 1): 726, (77, 2, 0): 574, (71, 7, 2): 500, (44, 4, 0): 288, (149, 1, 0): 1174, (77, 5, 0): 576, (33, 2, 0): 201, (23, 1, 5): 124, (40, 0, 0): 241, (52, 7, 0): 322, (13, 2, 5): 31, (57, 1, 0): 380, (107, 6, 0): 823, (73, 1, 0): 541, (32, 2, 0): 195, (139, 7, 0): 1091, (55, 4, 0): 358, (124, 2, 0): 984, (23, 4, 0): 133, (113, 1, 1): 860, (148, 2, 0): 1164, (28, 1, 4): 163, (86, 3, 0): 657, (86, 0, 0): 649, (119, 5, 0): 936, (90, 0, 0): 684, (48, 4, 1): 305, (23, 1, 2): 122, (124, 0, 2): 979, (75, 0, 2): 553, (71, 9, 5): 504, (40, 2, 0): 243, (113, 1, 4): 862, (133, 4, 5): 1044, (23, 4, 2): 135, (81, 4, 0): 614, (124, 2, 2): 985, (13, 3, 0): 32, (76, 0, 2): 560, (101, 1, 4): 766, (107, 0, 5): 805, (9, 0, 1): 14, (77, 1, 0): 572, (142, 2, 0): 1116, (113, 7, 2): 877, (77, 4, 0): 575, (151, 1, 0): 1204, (117, 0, 5): 913, (159, 0, 0): 1243, (48, 7, 0): 306, (53, 7, 2): 342, (107, 3, 5): 817, (64, 3, 0): 434, (86, 0, 4): 651, (99, 0, 0): 758, (139, 1, 4): 1085, (65, 3, 2): 460, (23, 4, 5): 136, (85, 7, 0): 647, (129, 0, 0): 1010, (112, 4, 0): 853, (72, 1, 1): 511, (79, 1, 4): 585, (120, 4, 0): 944, (115, 8, 0): 905, (43, 4, 4): 276, (124, 9, 0): 990, (43, 4, 0): 274, (150, 6, 0): 1197, (81, 7, 0): 619, (83, 7, 0): 634, (16, 6, 0): 68, (159, 7, 0): 1259, (75, 0, 0): 552, (147, 2, 0): 1141, (62, 0, 0): 413, (90, 2, 0): 685, (22, 1, 4): 103, (29, 6, 0): 180, (62, 1, 0): 414, (133, 1, 2): 1032, (18, 2, 0): 80, (43, 7, 4): 279, (57, 3, 0): 382, (149, 1, 6): 1177, (79, 0, 0): 579, (49, 0, 0): 308, (43, 7, 2): 278, (2, 0, 0): 2, (76, 1, 0): 562, (14, 1, 0): 54, (72, 0, 6): 509, (64, 4, 2): 438, (124, 10, 0): 991, (167, 0, 0): 1283, (159, 4, 1): 1256, (124, 0, 4): 980, (144, 1, 0): 1125, (45, 0, 0): 291, (107, 5, 4): 822, (115, 4, 0): 896, (52, 3, 0): 320, (53, 0, 4): 325, (123, 2, 0): 969, (79, 9, 0): 599, (118, 7, 0): 927, (15, 1, 0): 60, (13, 2, 2): 30, (91, 1, 0): 691, (50, 7, 0): 314, (122, 1, 4): 955, (95, 2, 0): 717, (154, 4, 0): 1226, (76, 3, 0): 565, (152, 9, 0): 1213, (141, 4, 0): 1109, (56, 2, 0): 369, (34, 1, 0): 202, (159, 0, 5): 1246, (119, 2, 0): 933, (136, 1, 0): 1065, (89, 9, 0): 682, (59, 4, 0): 401, (159, 9, 0): 1263, (64, 2, 5): 433, (65, 0, 2): 455, (119, 4, 0): 935, (133, 2, 5): 1038, (42, 4, 1): 257, (53, 5, 0): 340, (77, 0, 0): 571, (113, 6, 0): 875, (13, 1, 6): 28, (64, 9, 5): 452, (91, 0, 5): 690, (78, 1, 0): 578, (147, 7, 5): 1154, (99, 7, 0): 760, (42, 9, 0): 259, (81, 7, 4): 621, (79, 1, 1): 583, (72, 2, 5): 518, (58, 2, 0): 386, (44, 0, 0): 284, (89, 7, 4): 680, (116, 0, 0): 909, (91, 7, 0): 696, (133, 0, 5): 1030, (58, 1, 0): 385, (22, 9, 0): 114, (107, 3, 0): 816, (39, 4, 0): 240, (55, 1, 1): 354, (56, 1, 2): 366, (46, 3, 0): 294, (132, 0, 0): 1020, (71, 0, 2): 487, (139, 1, 2): 1084, (65, 2, 0): 458, (125, 2, 0): 994, (129, 3, 0): 1013, (94, 2, 0): 711, (13, 4, 4): 38, (21, 7, 0): 96, (46, 7, 0): 295, (122, 0, 4): 952, (39, 2, 0): 239, (147, 7, 0): 1151, (22, 3, 0): 106, (29, 0, 3): 169, (38, 0, 0): 216, (79, 6, 0): 594, (29, 0, 2): 168, (137, 3, 0): 1072, (44, 2, 0): 286, (61, 7, 0): 412, (64, 9, 0): 451, (133, 7, 2): 1048, (38, 7, 2): 232, (64, 1, 2): 427, (68, 0, 0): 471, (133, 2, 2): 1036, (109, 4, 0): 840, (59, 7, 0): 402, (23, 0, 0): 116, (95, 1, 0): 716, (107, 7, 2): 827, (38, 5, 0): 228, (153, 9, 0): 1223, (95, 3, 0): 718, (59, 0, 2): 392, (96, 3, 0): 734, (160, 7, 0): 1266, (147, 2, 2): 1143, (133, 1, 5): 1034, (14, 9, 0): 57, (149, 2, 0): 1178, (80, 0, 0): 601, (43, 8, 0): 280, (120, 1, 0): 941, (77, 1, 5): 573, (153, 7, 0): 1222, (18, 10, 0): 86, (17, 0, 0): 71, (64, 1, 1): 426, (113, 1, 2): 861, (144, 1, 5): 1126, (103, 7, 5): 789, (53, 4, 0): 336, (151, 0, 4): 1203, (91, 0, 4): 689, (14, 7, 0): 56, (123, 4, 0): 971, (23, 9, 5): 148, (147, 7, 2): 1152, (15, 9, 0): 63, (122, 10, 0): 964, (38, 0, 4): 218, (29, 4, 2): 178, (38, 7, 1): 231, (149, 7, 0): 1183, (64, 1, 4): 428, (89, 0, 4): 670, (113, 0, 5): 858, (150, 2, 5): 1192, (107, 1, 2): 808, (122, 0, 0): 950, (136, 2, 2): 1067, (72, 1, 4): 513, (76, 2, 2): 564, (56, 9, 5): 378, (72, 10, 0): 539, (22, 4, 0): 108, (124, 4, 0): 987, (113, 2, 2): 865, (150, 0, 5): 1188, (19, 10, 0): 88, (5, 0, 0): 8, (31, 7, 0): 192, (95, 0, 0): 715, (96, 1, 4): 727, (96, 1, 5): 728, (13, 4, 0): 35, (138, 9, 0): 1079, (18, 0, 4): 77, (132, 7, 0): 1025, (16, 7, 0): 69, (29, 7, 5): 183, (64, 3, 5): 436, (83, 9, 0): 635, (103, 2, 5): 782, (115, 0, 5): 887, (136, 0, 0): 1063, (155, 7, 0): 1233, (147, 7, 4): 1153, (117, 5, 0): 918, (123, 8, 0): 974, (87, 0, 0): 663, (79, 7, 4): 597, (115, 10, 0): 908, (53, 3, 0): 335, (53, 1, 6): 331, (166, 7, 0): 1282, (148, 4, 5): 1167, (64, 7, 4): 447, (64, 10, 0): 453, (167, 1, 4): 1285, (81, 9, 2): 624, (22, 2, 0): 104, (53, 4, 2): 337, (101, 2, 0): 767, (115, 7, 2): 903, (56, 1, 5): 368, (117, 9, 0): 921, (41, 1, 0): 247, (21, 1, 0): 93, (106, 1, 4): 798, (52, 0, 0): 317, (13, 1, 0): 23, (13, 3, 2): 33, (21, 0, 0): 91, (96, 0, 0): 720, (37, 0, 0): 214, (70, 7, 4): 485, (97, 0, 0): 751, (103, 1, 6): 779, (71, 0, 3): 488, (103, 1, 5): 778, (96, 5, 0): 739, (155, 2, 0): 1231, (84, 2, 0): 638, (72, 4, 5): 525, (43, 3, 2): 273, (118, 9, 0): 928, (115, 0, 2): 885, (128, 0, 2): 1005, (23, 1, 0): 120, (148, 7, 0): 1169, (121, 0, 0): 947, (94, 1, 0): 710, (58, 0, 0): 383, (85, 0, 5): 641, (139, 3, 2): 1089, (54, 1, 0): 348, (86, 2, 5): 656, (50, 4, 0): 313, (142, 4, 0): 1118, (109, 3, 0): 839, (35, 2, 0): 206, (122, 1, 2): 954, (25, 7, 0): 158, (111, 2, 0): 847, (39, 0, 0): 237, (43, 1, 4): 268, (34, 2, 0): 203, (59, 9, 0): 405, (101, 0, 0): 764, (151, 1, 2): 1205, (107, 1, 4): 809, (113, 8, 0): 879, (14, 2, 0): 55, (26, 7, 0): 159, (139, 1, 0): 1083, (66, 1, 0): 464, (56, 7, 2): 375, (12, 0, 0): 18, (53, 0, 2): 324, (13, 1, 2): 25, (117, 3, 5): 916, (85, 9, 0): 648, (52, 4, 0): 321, (129, 9, 0): 1015, (92, 1, 4): 700, (43, 7, 0): 277, (72, 7, 5): 533, (131, 0, 0): 1017, (18, 4, 0): 82, (46, 0, 0): 292, (83, 0, 0): 628, (113, 1, 5): 863, (97, 7, 0): 753, (159, 1, 5): 1250, (96, 0, 4): 723, (123, 9, 4): 976, (148, 2, 5): 1165, (47, 0, 0): 296, (55, 9, 0): 360, (139, 0, 4): 1082, (23, 7, 1): 140, (66, 7, 0): 467, (30, 0, 0): 185, (104, 0, 0): 794, (155, 1, 0): 1230, (16, 2, 0): 67, (107, 9, 0): 831, (134, 0, 0): 1055, (148, 6, 0): 1168, (141, 0, 4): 1102, (64, 5, 0): 441, (42, 1, 0): 250, (72, 5, 0): 526, (96, 8, 0): 745, (81, 6, 0): 618, (133, 10, 0): 1054, (64, 6, 0): 442, (23, 3, 5): 132, (6, 0, 1): 10, (81, 0, 4): 606, (36, 2, 0): 209, (107, 1, 1): 807, (155, 8, 0): 1234, (135, 1, 0): 1060, (165, 9, 0): 1280, (19, 0, 0): 87, (107, 6, 2): 824, (147, 3, 0): 1144, (119, 0, 0): 929, (147, 5, 4): 1149, (74, 9, 0): 550, (22, 0, 2): 98, (88, 1, 0): 665, (124, 1, 4): 983, (133, 8, 0): 1051, (115, 6, 0): 901, (40, 1, 0): 242, (106, 9, 0): 800, (159, 7, 4): 1261, (53, 4, 4): 338, (58, 9, 0): 389, (133, 9, 5): 1053, (107, 0, 3): 803, (113, 9, 2): 881, (15, 7, 0): 62, (86, 0, 2): 650, (15, 2, 0): 61, (75, 2, 0): 555, (167, 1, 0): 1284, (133, 4, 0): 1042, (72, 3, 0): 519, (101, 4, 0): 768, (141, 2, 0): 1106, (23, 0, 5): 119, (17, 1, 0): 72, (149, 2, 5): 1179, (162, 0, 0): 1269, (76, 0, 4): 561, (65, 7, 0): 461, (38, 2, 0): 223, (72, 7, 0): 530, (13, 5, 0): 40, (72, 9, 2): 536, (47, 1, 0): 297, (21, 3, 2): 95, (62, 2, 0): 415, (131, 7, 0): 1019, (6, 0, 0): 9, (57, 2, 0): 381, (53, 9, 5): 346, (99, 1, 0): 759, (154, 0, 0): 1224, (100, 0, 0): 761, (65, 0, 0): 454, (96, 10, 1): 750, (71, 4, 0): 498, (156, 1, 0): 1236, (96, 1, 0): 725, (107, 7, 5): 829, (167, 4, 0): 1288, (122, 3, 0): 957, (56, 4, 0): 373, (141, 0, 0): 1101, (23, 3, 0): 130, (61, 1, 0): 410, (150, 1, 5): 1190, (23, 5, 0): 137, (29, 0, 0): 167, (43, 0, 4): 264, (148, 0, 5): 1161, (138, 4, 0): 1077, (4, 0, 1): 7, (74, 1, 2): 545, (111, 0, 0): 844, (120, 3, 0): 943, (13, 7, 5): 47, (167, 2, 0): 1286, (103, 4, 0): 785, (103, 2, 2): 781, (21, 2, 0): 94, (79, 7, 0): 595, (38, 3, 0): 225, (125, 0, 0): 992, (28, 4, 0): 165, (120, 0, 2): 940, (22, 3, 5): 107, (83, 1, 0): 629, (112, 2, 0): 852, (74, 7, 0): 549, (164, 0, 0): 1273, (113, 7, 0): 876, (53, 9, 0): 345, (11, 0, 0): 17, (141, 7, 5): 1112, (70, 3, 0): 484, (115, 1, 4): 890, (38, 0, 2): 217, (23, 9, 2): 147, (13, 3, 5): 34, (64, 3, 2): 435, (79, 0, 2): 580, (62, 4, 0): 417, (23, 0, 2): 117, (43, 2, 0): 269, (23, 3, 2): 131, (42, 1, 1): 251, (143, 4, 0): 1122, (159, 4, 0): 1255, (81, 1, 0): 607, (133, 0, 0): 1027, (25, 4, 4): 157, (2, 1, 4): 3, (64, 1, 5): 429, (18, 1, 4): 79, (158, 4, 0): 1242, (115, 0, 4): 886, (93, 2, 0): 706, (149, 1, 4): 1175, (36, 1, 0): 208, (89, 3, 0): 676, (149, 0, 5): 1173, (13, 7, 2): 45, (129, 2, 0): 1012, (118, 1, 0): 923, (96, 4, 0): 736, (103, 0, 2): 773, (148, 4, 0): 1166, (53, 2, 2): 333, (119, 0, 5): 930, (150, 9, 0): 1200, (28, 2, 0): 164, (54, 9, 0): 350, (56, 1, 0): 365, (16, 1, 0): 66, (79, 4, 4): 591, (72, 7, 2): 531, (149, 3, 5): 1181, (103, 9, 1): 791, (127, 1, 0): 999, (59, 3, 2): 400, (96, 2, 2): 732, (101, 7, 0): 769, (153, 6, 0): 1221, (64, 8, 0): 449, (64, 2, 4): 432, (43, 2, 4): 271, (133, 0, 4): 1029, (56, 2, 5): 371, (155, 0, 0): 1229, (86, 2, 0): 655, (72, 9, 4): 537, (153, 3, 0): 1220, (56, 9, 0): 377, (86, 4, 0): 659, (149, 0, 2): 1172, (115, 1, 5): 891, (23, 6, 0): 138, (92, 4, 0): 703, (134, 4, 0): 1058, (43, 0, 3): 263, (85, 4, 0): 645, (117, 0, 0): 912, (81, 3, 2): 613, (22, 7, 2): 112, (53, 7, 0): 341, (36, 4, 0): 211, (121, 7, 0): 949, (72, 7, 4): 532, (43, 3, 0): 272, (69, 1, 0): 476, (122, 5, 0): 959, (82, 4, 0): 627, (96, 7, 0): 741, (38, 4, 0): 226, (16, 0, 0): 65, (38, 5, 4): 229, (38, 1, 1): 220, (83, 4, 0): 633, (113, 0, 2): 856, (71, 1, 4): 492, (147, 0, 0): 1132, (29, 1, 2): 173, (140, 3, 5): 1098, (13, 0, 5): 22, (107, 0, 4): 804, (13, 8, 0): 48, (107, 9, 2): 832, (115, 3, 0): 894, (113, 3, 0): 866, (72, 3, 5): 521, (156, 0, 0): 1235, (147, 9, 5): 1158, (120, 9, 0): 946, (9, 1, 4): 15, (84, 0, 0): 637, (48, 1, 0): 301, (32, 1, 0): 194, (136, 7, 2): 1069, (124, 0, 0): 978, (53, 7, 5): 344, (32, 9, 0): 200, (103, 2, 0): 780, (29, 0, 5): 170, (20, 0, 0): 89, (55, 1, 0): 353, (42, 0, 0): 248, (49, 1, 0): 309, (48, 0, 0): 299, (31, 2, 0): 190, (100, 1, 0): 762, (109, 2, 0): 838, (55, 7, 0): 359, (159, 3, 0): 1254, (29, 7, 1): 182, (65, 9, 0): 462, (138, 3, 0): 1076, (122, 7, 4): 962, (135, 0, 0): 1059, (13, 7, 4): 46, (85, 4, 5): 646, (36, 0, 0): 207, (127, 0, 0): 998, (139, 2, 0): 1086, (89, 3, 5): 677, (159, 1, 1): 1248, (115, 2, 2): 893, (72, 10, 5): 540, (15, 0, 5): 59, (103, 0, 0): 772, (0, 0, 0): 0, (58, 7, 0): 388, (159, 1, 4): 1249, (107, 2, 5): 815, (100, 3, 0): 763, (58, 4, 0): 387, (74, 0, 2): 543, (109, 1, 0): 837, (139, 4, 0): 1090, (42, 4, 0): 256, (124, 1, 2): 982, (122, 2, 0): 956, (103, 0, 5): 774, (53, 1, 4): 329, (74, 1, 0): 544, (142, 1, 0): 1115, (115, 4, 2): 897, (101, 1, 0): 765, (72, 1, 5): 514, (125, 3, 0): 995, (115, 1, 2): 889, (124, 7, 4): 989, (134, 1, 0): 1056, (63, 0, 0): 419, (138, 7, 0): 1078, (36, 3, 0): 210, (68, 2, 0): 472, (105, 0, 0): 795, (30, 2, 0): 186, (75, 6, 0): 556, (18, 9, 0): 85, (56, 0, 0): 362, (113, 0, 0): 855, (141, 3, 5): 1108, (59, 1, 0): 394, (119, 1, 0): 931, (13, 1, 5): 27, (59, 1, 4): 396, (70, 2, 0): 483, (43, 1, 2): 267, (82, 3, 0): 626, (106, 7, 0): 799, (76, 6, 0): 567, (74, 10, 0): 551, (23, 7, 4): 142, (78, 0, 0): 577, (146, 0, 0): 1130, (64, 7, 5): 448, (122, 1, 0): 953, (4, 0, 0): 6, (147, 5, 0): 1148, (44, 3, 0): 287, (59, 2, 2): 398, (107, 10, 0): 834, (64, 4, 5): 440, (141, 3, 0): 1107, (7, 0, 0): 11, (141, 1, 0): 1104, (38, 2, 2): 224, (14, 0, 0): 53, (22, 1, 0): 100, (107, 4, 0): 818, (93, 3, 0): 707, (69, 0, 0): 475, (50, 0, 0): 310, (89, 2, 0): 674, (53, 0, 5): 326, (147, 0, 5): 1135, (133, 6, 0): 1046, (103, 3, 5): 784, (22, 2, 5): 105, (59, 3, 0): 399, (159, 7, 1): 1260, (89, 9, 4): 683, (127, 7, 0): 1002, (113, 1, 0): 859, (38, 10, 0): 236, (96, 7, 1): 742, (23, 4, 1): 134, (44, 7, 0): 289, (159, 2, 4): 1253, (52, 2, 0): 319, (13, 9, 5): 51, (29, 7, 0): 181, (123, 7, 0): 973, (96, 10, 0): 749, (38, 1, 4): 222, (103, 10, 0): 793, (89, 1, 0): 672, (56, 7, 0): 374, (92, 0, 0): 698, (53, 2, 5): 334, (76, 0, 0): 559, (41, 0, 0): 246, (23, 2, 5): 129, (18, 1, 0): 78, (69, 7, 0): 479, (71, 2, 2): 494, (13, 6, 0): 42, (23, 1, 4): 123, (3, 0, 0): 4, (107, 9, 5): 833, (96, 9, 2): 747, (40, 4, 0): 244, (112, 1, 0): 850, (167, 3, 0): 1287, (32, 7, 0): 199, (141, 7, 0): 1110, (71, 1, 0): 490}  # noqa
  inv_tuple_map = {v: k for k, v in tuple_map.items()}
  args.input_size = tuple(int(x) for x in args.input_size.split(','))
  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)

  model = get_model(input_size=args.input_size + (3, ), backbone=args.backbone,
      weights=None)

  B5preds = []
  print(f'Loading weights {args.weights}')
  model.load_weights(args.weights)
  print(model.summary())
  row_id, target = [], []
  image_id_batch, img_batch = [], []
  for i in range(2):
    parquet_fn = os.path.join(root_path,f'train_image_data_{i}.parquet')
    df_full = pd.read_parquet(parquet_fn)
    print(df_full.shape)
    
    split_test = np.array_split(df_full, 5, axis = 0)
    for df in tqdm(split_test):
        image_ids = df['image_id'].values
        df = df.drop(['image_id'], axis=1)
        for k in range(len(image_ids)):
          image_id = image_ids[k]
          img = df.iloc[k].values
          img = normalize_image(img, org_width, org_height, args.input_size[1], args.input_size[0])
          img_batch.append(np.dstack([img] * 3))
          image_id_batch.append(image_id)
        
          if len(img_batch) >= args.batch_size:
            decoded = process_batch(image_id_batch, img_batch, row_id, target, model, inv_tuple_map)
            B5preds += decoded
            image_id_batch, img_batch = [], []

  # process remaining batch
  if len(img_batch) > 0:
    decoded = process_batch(image_id_batch, img_batch, row_id, target, model, inv_tuple_map)
    B5preds += decoded
    image_id_batch, img_batch = [], []

  return B5preds, row_id

B5preds, row_idB5 = main()
#print(B5preds)
print(np.array(B5preds).shape)
row_idB5[:50]


###############################
# SENET
###############################
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as torchtransforms
import cv2
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
modelpath = "./savedModels/se_resnext50_32x4d_fold2.pkl"

simple_transform_valid = torchtransforms.Compose([
    torchtransforms.ToTensor(),
    torchtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
class ClsTestDataset(Dataset):
    def __init__(self, df, torchtransforms):
        self.df = df
        self.pathes = self.df.iloc[:,0].values
        self.data = self.df.iloc[:, 1:].values
        self.torchtransforms = torchtransforms

    def __getitem__(self, idx):
        HEIGHT = 137
        WIDTH = 236
        #row = self.df.iloc[idx].values
        path = self.pathes[idx]
        img = self.data[idx, :]
        img = 255 - img.reshape(HEIGHT, WIDTH).astype(np.uint8)
        #img = crop_resize(img, size=128)
        #img = crop_resize(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)       
        img = torchtransforms.ToPILImage()(img)
        img = self.torchtransforms(img)
        return path, img
    def __len__(self):
        return len(self.df)

def make_loader(
        data_folder,
        batch_size=64,
        num_workers=2,
        is_shuffle = False,
):

    image_dataset = ClsTestDataset(df = data_folder,
                                    torchtransforms = simple_transform_valid)

    return DataLoader(
    image_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=is_shuffle
    )
    
#from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo
__all__ = ['SENet', 'se_resnext50_32x4d']
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):        
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    
def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    return model

model = se_resnext50_32x4d(pretrained=None)
model.avg_pool = nn.AdaptiveAvgPool2d(1)
model.last_linear = nn.Linear(model.last_linear.in_features, 186)
modelvalue = torch.load(modelpath, map_location='cuda:0')
newmodelvalue = {}
for kv in modelvalue:
    newmodelvalue[kv[4:]]=modelvalue[kv]        
model.load_state_dict(newmodelvalue)
#model.load_state_dict(modelvalue)
model = model.to(device)

def getmodeleval(model, dataloaders):
    model.eval()
    pathes=[]

    alllogit1 = []
    alllogit2 = []
    alllogit3 = []
    for path, img in dataloaders:
        img = img.to(device)
        pathes.extend(path)
        with torch.no_grad():
            output = model(img)
        logit1, logit2, logit3 = output[:,: 168],\
                                    output[:,168: 168+11],\
                                    output[:,168+11:]
        logit1 = F.softmax(logit1, dim=1).cpu().numpy() 
        logit2 = F.softmax(logit2, dim=1).cpu().numpy()
        logit3 = F.softmax(logit3, dim=1).cpu().numpy()
        alllogit1.extend(logit1.tolist())
        alllogit2.extend(logit2.tolist())
        alllogit3.extend(logit3.tolist())
    alllogit1 = np.array(alllogit1)
    alllogit2 = np.array(alllogit2)
    alllogit3 = np.array(alllogit3)
    
    print("getmodeleval::alllogit1.shape", alllogit1.shape)
    print("getmodeleval::alllogit2.shape", alllogit2.shape)
    print("getmodeleval::alllogit3.shape", alllogit3.shape)
    return pathes, alllogit1, alllogit2, alllogit3

import gc
allpathes=[]
allpreds_root = []
allpreds_vowel = []
allpreds_consonant = []
tAllBegin = time.time()
for i in range(2):
    test_csv = pd.read_parquet(os.path.join(root_path, f'train_image_data_{i}.parquet'))
    tBegin = time.time()
    split_test = np.array_split(test_csv, 5, axis = 0)
    
    for test_df in split_test:
    
        dataloaders = make_loader(data_folder = test_df,
                                               batch_size=8,
                                               num_workers = 0,
                                               is_shuffle = False)
        pathes, logit1, logit2, logit3 = getmodeleval(model, dataloaders)
        #preds_root = np.argmax(logit1, axis=1)
        #preds_vowel = np.argmax(logit2, axis=1)
        #preds_consonant = np.argmax(logit3, axis=1)

        allpathes.extend(pathes)
        allpreds_root.extend(logit1.tolist())
        allpreds_vowel.extend(logit2.tolist())
        allpreds_consonant.extend(logit3.tolist())
        
        del logit1, logit2, logit3
        gc.collect()
    del test_csv, split_test
    gc.collect()
        
    tEnd = time.time()
    print(i, int(round(tEnd * 1000)) - int(round(tBegin * 1000)), "ms")
tAllEnd = time.time()
print(len(allpathes), len(allpreds_root), len(allpreds_vowel), len(allpreds_consonant),  int(round(tAllEnd * 1000)) - int(round(tAllBegin * 1000)), "ms")

row_ids3=[]
se_net_preds=[]
for idx, image_id in enumerate(allpathes):
    se_net_preds.extend([allpreds_root[idx]])
    se_net_preds.extend([allpreds_vowel[idx]])
    se_net_preds.extend([allpreds_consonant[idx]])

    row_ids3.extend([str(image_id) + '_grapheme_root'])
    row_ids3.extend([str(image_id) + '_vowel_diacritic'])
    row_ids3.extend([str(image_id) + '_consonant_diacritic'])
    
###########################
# ENSEMBLE
##########################
targets = []
num_models = 2
for i in range(len(row_idB5)):
    # Set Prediction with average of 5 predictions
    sub_pred_value = np.argmax((se_net_preds[i] + B5preds[i]) / num_models)
    targets.append(sub_pred_value)

submission_df = pd.DataFrame({'row_id': row_idB5, 'target': targets})
submission_df    
