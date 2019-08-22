from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        print("ids ", ids)
        print("num_samples ", num_samples)
        print("ids ", ids)
        print("self.adj_info", self.adj_info)
		#只是截取取得需要C采样的节点对应的邻接信息，
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        # 让每一个节点的neighbor 也就是按照每一行随机化。
        print("adj_lists", adj_lists)		
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        print("adj_lists  ", adj_lists )
        #我们只会截取这些beigb 的数目
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        print("adj_lists  ", adj_lists )
        return adj_lists
