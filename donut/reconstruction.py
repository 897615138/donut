import tensorflow as tf
from tfsnippet.utils import TensorArgValidator

__all__ = ['masked_reconstruct', 'iterative_masked_reconstruct']


def masked_reconstruct(reconstruct, x, mask, validate_shape=True, name=None):
    """
    用重构输出替换' x '的隐元素。

    此方法可用于在' x '上执行缺失的数据赋值，重构输出为' x '。

    Args:
        reconstruct ((tf.Tensor) -> tf.Tensor): 重构x的方法
        x: 要用func重构的张量。
        mask: `int32` 必须可以广播成' x '的shape。标记每个元素是否被覆盖。
        validate_shape (bool): 是否要验证`mask`的形状
            (default :obj:`True`)
        name (str): 此操作在TensorFlow图中的名称。
            (default "masked_reconstruct")

    Returns:
        tf.Tensor: ' x '的隐元素被重构输出替换。
    """
    with tf.name_scope(name, default_name='masked_reconstruct'):
        # 将python的数据类型转换成TensorFlow可用的tensor数据类型
        x = tf.convert_to_tensor(x)  # type: tf.Tensor
        mask = tf.convert_to_tensor(mask, dtype=tf.int32)  # type: tf.Tensor

        # 针对x的广播掩码 广播机制依赖于数组shape属性
        old_mask = mask
        # 校验广播形状
        try:
            # 返回 shape_x 和 shape_y 之间的广播静态形状，进行广播
            _ = tf.broadcast_static_shape(x.get_shape(), mask.get_shape())
        except ValueError:
            raise ValueError('`mask`的形状不能广播到 `x`的形状 ({!r} vs {!r})'.
                             format(old_mask.get_shape(), x.get_shape()))
        # 使mask和x维度有一样的张量
        mask = mask * tf.ones_like(x, dtype=mask.dtype)

        # 是否要验证mask的形状
        if validate_shape:
            x_shape = x.get_shape()
            mask_shape = mask.get_shape()
            # 判断mask_shape和x_shape中的元素全部已知
            if mask_shape.is_fully_defined() and x_shape.is_fully_defined():
                # mask_shape与x_shape相同
                if mask_shape != x_shape:
                    # 唯一可能的情况是掩码的尺寸大于x，我们认为这种情况是无效的
                    raise ValueError('`mask`的形状不能广播到 `x`的形状({!r} vs {!r})'.format(old_mask.get_shape(), x_shape))
            # mask_shape和x_shape中的元素并非全部已知
            else:
                assert_op = tf.assert_equal(
                    # 因为已经被x * ones_like(x)广播了，我们只需要比较排名
                    tf.rank(x),
                    tf.rank(mask),
                    message='`mask`的形状不能广播到 `x`的形状'
                )
                with tf.control_dependencies([assert_op]):
                    mask = tf.identity(mask)

        # 获得重构 x
        r_x = reconstruct(x)

        # 根据mask获得输出
        return tf.where(tf.cast(mask, dtype=tf.bool), r_x, x)


def iterative_masked_reconstruct(reconstruct, x, mask, iter_count,
                                 back_prop=True, name=None):
    """
    用“mask”迭代地重构“x”“iter_count”次。

    这个方法将调用:func:`masked_reconstruct``iter_count`次，并将前一次迭代的输出作为下一次迭代的输入`x`。将返回最后一次迭代的输出。

    Args:
        reconstruct: 重构x的方法
        x: 被方法重构的张量
        mask: 32位整型，必须对x进行广播，指示每一个x是否要被覆盖掉
        iter_count (int or tf.Tensor):迭代次数 必须大于1
        back_prop (bool): 是否在所有迭代中支持反向传播?
            (default :obj:`True`)
        name (str): 此操作在TensorFlow图中的名称。
            (default "iterative_masked_reconstruct")

    Returns:
        tf.Tensor: 迭代重构的x。
    """
    with tf.name_scope(name, default_name='iterative_masked_reconstruct'):
        # 校验迭代次数
        v = TensorArgValidator('iter_count')
        iter_count = v.require_positive(v.require_int32(iter_count))

        # 覆盖重建
        x_r, _ = tf.while_loop(
            # 条件
            lambda x_i, i: i < iter_count,
            # 赋值 标记覆盖处重构
            lambda x_i, i: (masked_reconstruct(reconstruct, x_i, mask), i + 1),
            [x, tf.constant(0, dtype=tf.int32)],
            back_prop=back_prop
        )
        return x_r
