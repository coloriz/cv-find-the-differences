import cv2
import numpy as np
from pathlib import Path


def read_images(path, sz=None):
    import os
    import sys

    images = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            img_pair = []
            for filename in os.listdir(subject_path):
                try:
                    if filename == '.directory':
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename))
                    if im is None:
                        print("image " + filepath + " is none")
                    # resize to given size (if given)
                    if sz is not None:
                        im = cv2.resize(im, sz)
                    img_pair.append(im)
                except IOError as err:
                    errno, strerror = err.args
                    print("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            images.append(tuple(img_pair))

    return images


def harmonize_topologies(query, train):
    # convert image color to gray
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(query_gray, None)
    kp2, des2 = sift.detectAndCompute(train_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        r, c = train_gray.shape

        query_out = cv2.warpPerspective(query, M, dsize=(c, r), borderValue=(0, 0, 0))

        r, c = query_gray.shape
        pts = np.float32([[0, 0], [0, r - 1], [c - 1, r - 1], [c - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        roi_mask = np.zeros(train.shape, dtype=np.uint8)
        roi_corners = np.array([dst], dtype=np.int32)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        channel_count = train.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(roi_mask, roi_corners, ignore_mask_color)
        train_out = cv2.bitwise_and(train, roi_mask)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    return M, query_out, train_out


def maximum_rect(rectangles):
    min_x = min([x for x, y, w, h in rectangles])
    min_y = min([y for x, y, w, h in rectangles])
    width = max([x + w for x, y, w, h in rectangles]) - min_x
    height = max([y + h for x, y, w, h in rectangles]) - min_y

    return min_x, min_y, width, height


def cluster_rect(rectangles, distance):
    clustered_rectangles = []
    clustered = False
    while not clustered:
        # 사각형의 중심점
        center_points = [(x + w // 2, y + h // 2) for x, y, w, h in rectangles]
        # 첫 번째 사각형과 중심점에 대해서
        basis_rect = rectangles[0]
        basis_point = center_points[0]

        # 현재 선택된 점이 아닌 점
        other_points = [other_point for other_point in center_points[1:]]
        # 현재 선택된 사각형이 아닌 사각형
        other_rects = [other_rect for other_rect in rectangles[1:]]
        cluster = [basis_rect]   # 클러스터를 저장할 리스트
        # 현재 기준점과 다른 점들과 거리를 비교
        for other_rect, other_point in zip(other_rects, other_points):
            x_basis, y_basis = basis_point
            x_point, y_point = other_point
            if ((x_basis - x_point) ** 2 + (y_basis - y_point) ** 2) <= distance ** 2:
                cluster.append(other_rect)

        max_rect = maximum_rect(cluster)
        # 만약 cluster가 1이라면 그 cluster는 최종 cluster
        if len(cluster) == 1:
            clustered_rectangles.append(max_rect)
            rectangles.remove(cluster[0])
        # 사각형 리스트를 갱신
        else:
            for rect in cluster:
                rectangles.remove(rect)
            rectangles.append(max_rect)
        if len(rectangles) == 1:
            clustered_rectangles.append(rectangles[0])
            clustered = True

    return clustered_rectangles


def points_to_bounding_rect(points):
    min_x = int(min([x for x, y in points]))
    min_y = int(min([y for x, y in points]))
    width = int(max([x for x, y in points])) - min_x
    height = int(max([y for x, y in points])) - min_y

    return min_x, min_y, width, height


imgs = read_images('./images')
for i, (a_original, b_original) in enumerate(imgs):
    M, _b, _a = harmonize_topologies(b_original, a_original)
    a, b = _a.copy(), _b.copy()

    a, b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a, b = a / 255, b / 255
    a, b = cv2.normalize(a, a, 1, 0, cv2.NORM_INF), cv2.normalize(b, b, 1, 0, cv2.NORM_INF)

    c = np.abs(a - b)
    c = cv2.GaussianBlur(c, (13, 13), 0)
    # cv2.imshow('|a - b|', c)

    _, c = cv2.threshold(c, 0.15, 1, cv2.THRESH_BINARY)
    c = cv2.erode(c, np.ones((2, 2), dtype=np.uint8), iterations=2)
    c = cv2.dilate(c, np.ones((2, 2), dtype=np.uint8), iterations=5)
    c = (c * 255).astype(np.uint8)
    # cv2.imshow('c_threshold', c)

    _, contours, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 200]

    # unclustered_a_original = a_original.copy()
    # for x, y, w, h in rectangles:
    #    cv2.rectangle(unclustered_a_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    # cv2.imshow('unclustered', unclustered_a_original)

    clustered_a_original = a_original.copy()
    clustered_rect = cluster_rect(rectangles, 60)
    for x, y, w, h in clustered_rect:
        cv2.rectangle(clustered_a_original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow('clustered A', clustered_a_original)

    M_inv = np.linalg.inv(M)
    clustered_b_original = b_original.copy()
    clustered_rect2 = []
    for x, y, w, h in clustered_rect:
        pts = np.float32([(x, y), (x, y + h - 1), (x + w - 1, y + h - 1), (x + w - 1, y)]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M_inv)
        clustered_rect2.append(points_to_bounding_rect(dst[:, 0]))

    for x, y, w, h in clustered_rect2:
        cv2.rectangle(clustered_b_original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow('clustered B', clustered_b_original)

    print(f'================================ {i} ===============================')
    for (x1, y1, w1, h1), (x2, y2, w2, h2) in zip(clustered_rect, clustered_rect2):
        print(f'A : ({x1}, {y1}), {w1 * h1}     B : ({x2}, {y2}), {w2 * h2}')


    my_path = Path('./result')
    if not my_path.is_dir():
        my_path.mkdir()

    a_height, a_width, _ = a_original.shape
    b_height, b_width, _ = b_original.shape
    result_height = max(a_height, b_height)
    result_width = a_width + b_width
    result = np.zeros((result_height, result_width, 3), dtype=np.uint8)
    result[0:a_height, 0:a_width, :] = clustered_a_original
    result[0:b_height, a_width:, :] = clustered_b_original
    cv2.imshow('result', result)
    cv2.imwrite(f'./result/sol{i}.jpg', result)

    cv2.waitKey()
