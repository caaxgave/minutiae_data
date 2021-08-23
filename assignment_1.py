import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET 

from cmath import rect, phase
from math import radians, degrees

from scipy.spatial import ConvexHull

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import networkx as nx
from itertools import combinations

"""## Shared Functions"""

def extract_features(Values, angle=False):
  sorted_values = np.sort(Values)
  
  min_value = min(Values)
  max_value = max(Values)
  avrg_value = np.average(Values)
  variance_value = np.var(Values)
  median_value = np.median(Values)
  #std_value = np.std(Values)
  minmax_ratio = min_value/max_value

  if angle:
    avrg_value = degrees(phase(sum(rect(1, radians(d)) for d in Values)/len(Values)))

  second_min = sorted_values[1]
  second_max = sorted_values[len(Values)-2]

  return [min_value, max_value, avrg_value, variance_value, median_value, minmax_ratio, second_min, second_max]

def distances_fromPoint(fingerprint, reference):
  distances = []
  for idx in range(len(fingerprint)):
    euclidian_distance = np.sqrt(((fingerprint[idx][1] - reference[0])**2) + ((fingerprint[idx][2] - reference[1])**2))
    distances.append(euclidian_distance)
  return distances

def distances_mnht_fromPoint(fingerprint, reference):
  distances = []
  for idx in range(len(fingerprint)):
    manhatan_distance = abs(fingerprint[idx][1] - reference[0]) + abs(fingerprint[idx][2] - reference[1])
    distances.append(manhatan_distance)
  return distances

def features_inside_radius(Values, radius):
  vals = np.array(Values)
  inside = []
  for idx in radius:
    minutiae_inside = vals[vals<=idx]
    if minutiae_inside.shape[0] == 0:
      inside.append(0)
      inside.append(0)
    else:
      inside.append(np.average(minutiae_inside))
      inside.append(minutiae_inside.shape[0])

  return inside

def near_far_minutia(fingerprint, reference_indx):
  minutiae = np.delete(fingerprint, reference_indx, 0)
  ref_minutia = (fingerprint[reference_indx][1], fingerprint[reference_indx][2])
  distan_minutiae = distances_fromPoint(minutiae, ref_minutia)
  near = np.argmin(distan_minutiae)
  far = np.argmax(distan_minutiae)
  if near >= reference_indx:
    near += 1
  if far >= reference_indx:
    far += 1

  return [near, far]

def near_far_features(fingerprint, angle):
  near = []
  far = []

  for idx in range (len(fingerprint)):
    nearfar_idx = near_far_minutia(fingerprint, idx)
    if angle == "beta":
      near.append(compute_beta(fingerprint[idx], fingerprint[nearfar_idx[0]]))
      far.append(compute_beta(fingerprint[idx], fingerprint[nearfar_idx[1]]))
    if angle == "alpha":
      near.append(compute_alpha(fingerprint[idx], fingerprint[nearfar_idx[0]]))
      near.append(compute_alpha(fingerprint[nearfar_idx[0]], fingerprint[idx]))

      far.append(compute_alpha(fingerprint[idx], fingerprint[nearfar_idx[1]]))
      far.append(compute_alpha(fingerprint[nearfar_idx[1]], fingerprint[idx]))

  return [np.average(near), np.var(near), np.average(far), np.var(far)]

def feature_variation(data_before, data_after):
  variation = []
  for idx in range(len(data_before)):
    variation.append(data_after[idx] - data_before[idx])

  return variation

"""## Distances attributes"""

def distances_arrays(fingerprint): 
  euclidian_distances = []
  manhatan_distances = []
  for idx in range(len(fingerprint)-1):
    for idy in range(idx+1, len(fingerprint)):
      euclidian = np.sqrt(((fingerprint[idx][1] - fingerprint[idy][1]) ** 2) + ((fingerprint[idx][2] - fingerprint[idy][2]) ** 2))
      manhatan = abs(fingerprint[idx][1] - fingerprint[idy][1]) + abs(fingerprint[idx][2] - fingerprint[idy][2])
      euclidian_distances.append(euclidian)
      manhatan_distances.append(manhatan)
  
  return euclidian_distances, manhatan_distances


def distances(fingerprint):
  both_distances = distances_arrays(fingerprint)

  return extract_features(both_distances[0])[:6] + extract_features(both_distances[1])[:6]

def target_distances(fingerprint, target_minu):
  target_location = (fingerprint[target_minu][1], fingerprint[target_minu][2])
  distances_eu = distances_fromPoint(np.delete(fingerprint, target_minu, 0), target_location)
  distances_mnht = distances_mnht_fromPoint(np.delete(fingerprint, target_minu, 0), target_location)

  return extract_features(distances_eu)[:6] + extract_features(distances_mnht)[:6]

"""## Beta attributes"""

def beta(fingerprint):
  beta_angles = []
  
  for idx in range(len(fingerprint)-1):
    for idy in range(idx+1, len(fingerprint)):
      beta_angles.append(compute_beta(fingerprint[idx], fingerprint[idy]))
  
  return extract_features(beta_angles, angle=True) + near_far_features(fingerprint, "beta")

def target_beta(fingerprint, target):
  beta_angles = []
  target_point = (fingerprint[target][1], fingerprint[target][2])
  new_set = np.delete(fingerprint, target, 0)
  for idx in range(len(new_set)):
    beta_angles.append(compute_beta(fingerprint[target], new_set[idx]))
  distances_ft = distances_fromPoint(new_set, target_point)

  beta_1nn = beta_angles[np.argmin(distances_ft)]
  beta_2nn = beta_angles[np.argsort(distances_ft)[1]]

  beta_1fn = beta_angles[np.argmax(distances_ft)]
  beta_2fn = beta_angles[np.argsort(distances_ft)[-2]]

  return extract_features(beta_angles, angle=True) + [beta_1nn, beta_2nn, beta_1fn, beta_2fn]

def compute_beta(point_i, point_j):
  b_angle = min(abs(point_i[3] - point_j[3]), (360 - abs(point_i[3] - point_j[3])))
  return b_angle

"""## Alpha angles"""

def alpha(fingerprint):
  alpha_angles = []
  for idx in range(len(fingerprint)-1):
    for idy in range(idx+1, len(fingerprint)):
      alpha_ij = compute_alpha(fingerprint[idx], fingerprint[idy])
      alpha_ji = compute_alpha(fingerprint[idy], fingerprint[idx])
      alpha_angles.append(alpha_ij)
      alpha_angles.append(alpha_ji)

  return extract_features(alpha_angles, angle=True) + near_far_features(fingerprint, "alpha")

def target_alpha(fingerprint, target):
  alpha_angles = []
  target_point = (fingerprint[target][1], fingerprint[target][2])
  new_set = np.delete(fingerprint, target, 0)
  for idx in range(len(new_set)):
    alpha_angles.append(compute_alpha(fingerprint[target], new_set[idx]))
  distances_ft = distances_fromPoint(new_set, target_point)

  alpha_1nn = alpha_angles[np.argmin(distances_ft)]
  alpha_2nn = alpha_angles[np.argsort(distances_ft)[1]]

  alpha_1fn = alpha_angles[np.argmax(distances_ft)]
  alpha_2fn = alpha_angles[np.argsort(distances_ft)[-2]]

  return extract_features(alpha_angles, angle=True) + [alpha_1nn, alpha_2nn, alpha_1fn, alpha_2fn]

def compute_alpha(point_i, point_j):
  ang = ang_angle(point_i, point_j)
  alpha_angle = min(abs(point_i[3] - ang), (360-abs(point_i[3] - ang)))

  return alpha_angle

def ang_angle(minutia_i, minutia_j):
  delta_x = minutia_j[1] - minutia_i[1]
  delta_y = minutia_j[2] - minutia_i[2]

  if (delta_x > 0 and delta_y >= 0):
    ang = np.arctan(delta_y/delta_x)
    return ang

  if (delta_x > 0 and delta_y < 0):
    ang = np.arctan(delta_y/delta_x) + 360
    return ang

  if delta_x < 0:
    ang = np.arctan(delta_y/delta_x) + 180
    return ang

  if delta_x == 0 and delta_y > 0:
    ang = 90
    return ang

  if delta_x == 0 and delta_y < 0:
    ang = 270
    return ang

"""## Centroid"""

def centroid(fingerprint):
  sum_x = np.sum(fingerprint[:,1])
  sum_y = np.sum(fingerprint[:,2])
  set_centroid = (sum_x/fingerprint.shape[0], sum_y/fingerprint.shape[0])
  centroid_distances = distances_fromPoint(fingerprint, set_centroid)
  radius = [30,45,60]
  centroid_inside = features_inside_radius(centroid_distances, radius)
  
  return extract_features(centroid_distances)[:6] + centroid_inside

def target_centroid(fingerprint, target):
  sum_x = np.sum(fingerprint[:,1])
  sum_y = np.sum(fingerprint[:,2])
  set_centroid = (sum_x/fingerprint.shape[0], sum_y/fingerprint.shape[0])
  euclidian_center = np.sqrt(((set_centroid[0] - fingerprint[target][1])**2) + ((set_centroid[1] - fingerprint[target][2])**2))
  manhatan_center = abs(set_centroid[0] - fingerprint[target][1]) + abs(set_centroid[1] - fingerprint[target][2])

  target_centroid_distances = distances_fromPoint(np.delete(fingerprint, target, 0), (fingerprint[target][1], fingerprint[target][2]))
  radius = [30,45,60,95]
  direction = ang_angle([0, set_centroid[0], set_centroid[1], 0], fingerprint[target])

  fingerprint_nt = np.delete(fingerprint, target, 0)
  centroid_nt = (np.sum(fingerprint_nt[:,1])/fingerprint_nt.shape[0], np.sum(fingerprint_nt[:,2])/fingerprint_nt.shape[0])
  euclidian_center_nt = np.sqrt(((centroid_nt[0] - fingerprint[target][1])**2) + ((centroid_nt[1] - fingerprint[target][2])**2))

  return [euclidian_center, manhatan_center, euclidian_center_nt, direction] + features_inside_radius(target_centroid_distances, radius)

"""## Convex Hull"""

def convex_hull(fingerprint):
  points = fingerprint[:,1:3]
  hull = ConvexHull(points)
  convex_points = hull.vertices
  
  perimeter = hull.area
  area = hull.volume
  minutiae_convex = len(convex_points)
  minutiae_inside = len(fingerprint) - minutiae_convex
  
  return [perimeter, area, minutiae_convex, minutiae_inside] + convex_distances(np.array(fingerprint)[convex_points])

def convex_distances(points_convex):
  distances = []
  for idx in range(len(points_convex)-1):
    euclidian = np.sqrt(((points_convex[idx][1] - points_convex[idx+1][1])**2) + ((points_convex[idx][2] - points_convex[idx+1][2])**2) )
    distances.append(euclidian)
  distances.append(np.sqrt(((points_convex[len(points_convex)-1][1] - points_convex[0][1])**2) + ((points_convex[len(points_convex)-1][2] - points_convex[0][2])**2) ))

  return extract_features(distances)

"""## MST"""

def graph_maker(fingerprint):
  minutiae = [i for i in range(len(fingerprint))] #nodes
  distances = distances_arrays(fingerprint)[0] #edges268

  combinations_list = list(combinations(minutiae,2))
  matrix = np.zeros((len(minutiae),len(minutiae)), dtype=int)
  i=0
  for node in combinations_list:
    matrix[node[0]][node[1]] = distances[i]
    i += 1
  
  return matrix.tolist()

def max_branches(mst):
  list1 = list(np.where(mst != 0))[0].tolist()
  list2 = list(np.where(mst != 0))[1].tolist()
  whole_list = list1 + list2 
  index = max(whole_list, key=whole_list.count)
  max_branches = whole_list.count(index)

  first_max_branch_distance = 0
  for i in range(len(mst[0])):
     first_max_branch_distance += mst[index][i] + mst[i][index]

  return [first_max_branch_distance, max_branches]


def number_leafs(mst):
  j, leafs = 0, 0
  list1 = list(np.where(mst != 0))[0].tolist()
  list2 = list(np.where(mst != 0))[1].tolist()
  whole_list = list1 + list2
  for i in range(len(mst[0])):
    if whole_list.count(i) == 1:
      leafs += 1

  return leafs 

def min_spanning_tree(fingerprint):
  graph = csr_matrix(graph_maker(fingerprint))
  mst = minimum_spanning_tree(graph)

  total_distance = np.sum(mst)
  mst = mst.toarray().astype(int)

  return [total_distance, number_leafs(mst)] + max_branches(mst) + extract_features(np.array(mst)[mst > 0])



