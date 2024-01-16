import numpy as np
import torch


# 用于计算混淆矩阵
def fast_hist(pred, label, n):
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    k = (label >= 0) & (pred < n)
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，斜对角线上的为分类正确的像素点
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    return np.diag(hist) / (np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) + 1e-6)


def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


# 用混淆矩阵计算IOU
def compute_mIoU(pred, label, num_classes):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    # 计算混淆矩阵，输入要求展开成一维
    hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
    print(hist)
    #   计算所有验证集图片的逐类别mIoU值
    per_iou = per_class_iou(hist)
    mPA = per_class_PA(hist)
    # 在所有验证集图像上求所有类别平均的mIoU值，不包括背景部分(0)
    per_iou = per_iou[1:]
    mIoUs = np.nanmean(per_iou)
    mPA = np.nanmean(mPA)
    return mIoUs


# 一种计算iou和dice的方法
def iou_and_dice(output, target):
    smooth = 1e-6

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = intersection / (union + smooth)
    # 根据dice和iou的关系计算dice
    dice = (2 * iou) / (iou + 1)
    return iou, dice


# 一种计算dice coef的方法，计算前先进行了一次sigmoid，使得计算出的dice更稳定，但与定义有差异
def dice_coef(output, target):
    smooth = 1e-6

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection) / \
        (output.sum() + target.sum() + smooth)


# 一种官方给出的计算dice的方法，适用于batch为1的时候用
def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / (float(size_i1 + size_i2) + 1e-6)
    except ZeroDivisionError:
        dc = 0.0

    return dc


# 一种基于官方方法改进的计算dice的方法，可以计算batch≥1时的平均dice
def dc_mean(output, target):
    dc_list = []
    # 转换数据类型，方便进行计算
    output = output > 0.5
    target = target > 0.5
    output = output.cpu().numpy()
    target = target.cpu().numpy()
    for i in range(output.shape[0]):
        output_i = output[i, :, :, :]
        target_i = target[i, :, :, :]
        output_i = np.atleast_1d(output_i.astype(np.bool))
        target_i = np.atleast_1d(target_i.astype(np.bool))

        intersection = np.count_nonzero(output_i & target_i)

        size_i1 = np.count_nonzero(output_i)
        size_i2 = np.count_nonzero(target_i)

        try:
            dc = 2. * intersection / (float(size_i1 + size_i2) + 1e-6)
        except ZeroDivisionError:
            dc = 0.0
        finally:
            dc_list.append(dc)
    dc = np.mean(dc_list)
    return dc


# Jaccard Coefficient
def jc(result, reference):
    """
    Jaccard coefficient

    Computes the Jaccard coefficient between the binary objects in two images.

    Parameters
    ----------
    result: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    reference: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.

    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `result` and the
        object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    union = np.count_nonzero(result | reference)

    jc = float(intersection) / (float(union) + 1e-6)

    return jc


def recall(result, reference):
    """
    Recall.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.

    See also
    --------
    :func:`precision`

    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / (float(tp + fn) + 1e-6)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def ACC(result, reference):
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference)

    try:
        ACC = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        ACC = 0.0

    return ACC


def ravd(result, reference):
    """
    Relative absolute volume difference.

    Compute the relative absolute volume difference between the (joined) binary objects
    in the two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    ravd : float
        The relative absolute volume difference between the object(s) in ``result``
        and the object(s) in ``reference``. This is a percentage value in the range
        :math:`[-1.0, +inf]` for which a :math:`0` denotes an ideal score.

    Raises
    ------
    RuntimeError
        If the reference object is empty.

    See also
    --------
    :func:`dc`
    :func:`precision`
    :func:`recall`

    Notes
    -----
    This is not a real metric, as it is directed. Negative values denote a smaller
    and positive values a larger volume than the reference.
    This implementation does not check, whether the two supplied arrays are of the same
    size.

    Examples
    --------
    Considering the following inputs

    >>> import numpy
    >>> arr1 = numpy.asarray([[0,1,0],[1,1,1],[0,1,0]])
    >>> arr1
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> arr2 = numpy.asarray([[0,1,0],[1,0,1],[0,1,0]])
    >>> arr2
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])

    comparing `arr1` to `arr2` we get

    >>> ravd(arr1, arr2)
    -0.2

    and reversing the inputs the directivness of the metric becomes evident

    >>> ravd(arr2, arr1)
    0.25

    It is important to keep in mind that a perfect score of `0` does not mean that the
    binary objects fit exactely, as only the volumes are compared:

    >>> arr1 = numpy.asarray([1,0,0])
    >>> arr2 = numpy.asarray([0,0,1])
    >>> ravd(arr1, arr2)
    0.0

    """
    # 转换数据类型，方便进行计算
    result = result.cpu().numpy()
    reference = reference.cpu().numpy()

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return (vol1 - vol2) / float(vol2)
