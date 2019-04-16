import os

import numpy as np
import tensorflow as tf
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Seq2Point:
    def __init__(self,table_name="", checkpoint_dir="",
                 epochs=100, batch_size=500, seq_length=100, feature_dim=56, term_length = 15, item_features_dim = 34,
                 query_features_dim=7, m1=1, m2=1, m3=1, is_train=True, is_debug = True):

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.001

        # IO config
        self.table_name = table_name
        self.checkpointDir = checkpoint_dir
        # param config
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.query_features_dim = query_features_dim
        self.item_features_dim = item_features_dim
        self.output_length = 1
        self.feature_dim = feature_dim
        self.term_length = term_length
        self.is_train = is_train
        self.is_debug = is_debug

        # model_param
        self.rnn_cell_dim = 256
        self.dense_dim = 50#128
        self.attention_dim = 10

        self.short_dim = 4

        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.X = []
        self.X_last = ''
        self.brand_ids = ''
        self.fc1 = ''
        self.fc2 = ''
        self.embed_tmp1 = ''
        self.embed_tmp2 = ''
        self.X1 = ''
        self.X2 = ''

    # Init Graph
    def build_input(self):
        self.Ruserid, self.Rcate_list, self.Rcatelevel1_list, self.Rbrand_list, self.Ritem_list, self.Rtime_list, self.Raction_list, self.Ritem_name_hash_list, self.Ri2q_term_hash_list, self.Ritem_features_list, self.Rquery_hash_list, self.Rquery_features, self.Rqid, self.Rcateid1, self.Rcateid2, self.Rcateid3, self.Rscore_features, self.Rxftrl_features, self.Rmatchtype, self.Rcateid, self.Rtriggerid, self.Rage_class, self.Rbaby_stage, self.Rcareer_type, self.label_value, self.label_oh = self.__data__()

    def build_graph(self):
        self.pred_logit = self.__build_graph__()
        self.loss, self.optimizer = self.__loss_optimizer__()

    def build_summary(self, name='train'):
        pset, mset = self.__evaluation__()
        self.positive_score, self.pred_binary = pset
        self.acc, self.precision, self.recall, self.auc = mset
        self.summary_op = self.__add_summary__(name)

    def __add_summary__(self, name):
        print('summary')
        summary = [
            tf.summary.scalar(name + '/loss', self.loss),
            tf.summary.scalar(name + '/metrics/acc', self.acc),
            tf.summary.scalar(name + '/metrics/precision', self.precision),
            tf.summary.scalar(name + '/metrics/recall', self.recall),
            tf.summary.scalar(name + '/metrics/auc', self.auc),
            # tf.summary.histogram(name+'/attention', self.att_value),
            # tf.summary.histogram(name+'/final_out', self.final_out),
        ]
        summary_op = tf.summary.merge(summary)
        return summary_op

    def __evaluation__(self):
        print('evaluation')
        positive_score = tf.slice(self.pred_logit, [0, 1], [-1, 1])
        pred_binary = tf.cast(tf.round(positive_score), tf.int32)

        _, acc = tf.metrics.accuracy(self.label_value, pred_binary)
        _, precision = tf.metrics.precision(self.label_value, pred_binary)
        _, recall = tf.metrics.recall(self.label_value, pred_binary)
        _, auc = tf.metrics.auc(self.label_value, positive_score)
        return (positive_score, pred_binary), (acc, precision, recall, auc)

    def __train__(self, loss, optimizer=None):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(loss, global_step=self.global_step)

    def __loss_optimizer__(self):
        print('make loss')
        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label_oh, logits=self.pred_logit)
        if self.is_train:
            optimizer = self.__train__(loss)
            return loss, optimizer
        else:
            return loss, None

    def __build_graph__(self):
        print('Build Graph')

        with tf.variable_scope('embedding', initializer=tf.zeros_initializer()):
            print 'self.m1,self.m2,self.m3:',self.m1,self.m2,self.m3
            if self.m1 == "1":
                print 'm1 is activated'
            self.embedding_catelevel1id = tf.get_variable('catelevel1_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_brandid = tf.get_variable('brand_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_itemid = tf.get_variable('item_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_actionid = tf.get_variable('action_embedding', [100, self.rnn_cell_dim * 2], tf.float32, tf.random_normal_initializer())
            self.embedding_termid = tf.get_variable('term_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_userid = tf.get_variable('user_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_queryid = tf.get_variable('query_embedding', [100000, self.query_features_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_matchtype = tf.get_variable('matchtype_embedding', [100000, self.short_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_cateid = tf.get_variable('cate_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_triggerid = tf.get_variable('triggerid_embedding', [100000, self.feature_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_age_class = tf.get_variable('age_class_embedding', [100000, self.short_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_baby_stage = tf.get_variable('baby_stage_embedding', [100000, self.short_dim], tf.float32, tf.random_normal_initializer())
            self.embedding_career_type = tf.get_variable('career_type_embedding', [100000, self.short_dim], tf.float32, tf.random_normal_initializer())

            print "self.Rcatelevel1_list.shape", self.Rcatelevel1_list.shape 

            catelevel1_emb = emb_from_id(self.Rcatelevel1_list, self.embedding_catelevel1id, [self.seq_length, self.batch_size, self.feature_dim]) 
            brand_emb = emb_from_id(self.Rbrand_list, self.embedding_brandid, [self.seq_length, self.batch_size, self.feature_dim])
            action_emb = tf.nn.embedding_lookup(self.embedding_actionid, self.Raction_list%100)
            item_name_hash_emb = emb_from_id(self.Ritem_name_hash_list, self.embedding_termid, [self.seq_length, self.term_length, self.batch_size, self.feature_dim])
            i2q_term_hash_emb = emb_from_id(self.Ri2q_term_hash_list, self.embedding_termid, [self.seq_length, self.term_length, self.batch_size, self.feature_dim])
            query_hash_emb = emb_from_id(self.Rquery_hash_list, self.embedding_termid, [self.term_length, self.batch_size, self.feature_dim])
            user_emb = emb_from_id(self.Ruserid, self.embedding_userid, [self.batch_size, self.feature_dim])
            query_emb = emb_from_id(self.Rqid, self.embedding_queryid, [self.batch_size, self.query_features_dim])
            cate1_emb = emb_from_id(self.Rcateid1, self.embedding_catelevel1id, [self.batch_size, self.feature_dim])
            cate2_emb = emb_from_id(self.Rcateid2, self.embedding_catelevel1id, [self.batch_size, self.feature_dim])
            cate3_emb = emb_from_id(self.Rcateid3, self.embedding_catelevel1id, [self.batch_size, self.feature_dim])
            matchtype_emb = emb_from_id(self.Rmatchtype, self.embedding_matchtype, [self.batch_size, self.short_dim])
            cate_emb = emb_from_id(self.Rcateid, self.embedding_cateid, [self.batch_size, self.feature_dim])
            trigger_emb = emb_from_id(self.Rtriggerid, self.embedding_triggerid, [self.batch_size, self.feature_dim])
            age_class_emb = emb_from_id(self.Rage_class, self.embedding_age_class, [self.batch_size, self.short_dim])
            baby_stage_emb = emb_from_id(self.Rbaby_stage, self.embedding_baby_stage, [self.batch_size, self.short_dim])
            career_type_emb = emb_from_id(self.Rcareer_type, self.embedding_career_type, [self.batch_size, self.short_dim])

            self.Ritem_features_list = tf.transpose(self.Ritem_features_list, perm=[0,2,1])
            self.Rscore_features = tf.transpose(self.Rscore_features, perm=[1,0])
            self.Rxftrl_features = tf.transpose(self.Rxftrl_features, perm=[1,0])
            self.Rquery_features = tf.transpose(self.Rquery_features, perm=[1,0])

            #X = tf.concat([catelevel1_emb, brand_emb, tf.reduce_mean(item_name_hash_emb, 1), tf.reduce_mean(i2q_term_hash_emb, 1), self.Ritem_features_list], axis=2)
            X = tf.concat([catelevel1_emb, tf.reduce_mean(item_name_hash_emb, 1), tf.reduce_mean(i2q_term_hash_emb, 1)], axis=2)
            #self.X_last = tf.concat([self.Rquery_features, cate1_emb + cate2_emb + cate3_emb, tf.reduce_mean(query_hash_emb, 0)], axis=1)
            self.X_last = tf.concat([(cate1_emb + cate2_emb + cate3_emb)/3, tf.reduce_mean(query_hash_emb, 0)], axis=1)
            #self.X_last = tf.concat([(cate1_emb + cate2_emb + cate3_emb)/3], axis=1)

            self.X = tf.unstack(X, axis=0)
            self.Xone = self.X[0]
            self.X1 = self.X[-1]
            self.X2 = self.X[-2]


        #with tf.device('/cpu:0'):
            #####       Encoder Level       #####
            
            fw_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_cell_dim)
            bw_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_cell_dim)
#            if self.is_train:
#                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.9)
#                bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.9)

            bi_output, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, self.X,
                                                                                    dtype=tf.float32)
            #####       Attention Level       #####
            # 19*cell
            outputs = tf.stack(bi_output)  # L, N, D
            outputs = tf.multiply(outputs,action_emb) # multyply by action
            #incorporate the time_decay into the attention design
            self.time_decay = tf.get_variable('time_decay', initializer = tf.stack([-0.2]))
            self.time_decay = tf.clip_by_value(self.time_decay, -0.5, 0)
            decay_expand = tf.expand_dims(self.time_decay, -1)
            decay_tile = tf.tile(decay_expand, self.Rtime_list.shape)
            time_decay_pow = tf.pow(self.Rtime_list, decay_tile)
            time_decay_pow_expand = tf.expand_dims(time_decay_pow, -1)
            time_decay_pow_tile = tf.tile(time_decay_pow_expand, [1, 1, outputs.shape[2]])
            outputs = tf.multiply(outputs, time_decay_pow_tile)
            print 'end time_decay'
            #end time_decay
            state = tf.concat([state_fw, state_bw, self.X_last], 1)  # N, 3*D
            attention_type = 'nn'#'bilinear'
            with tf.variable_scope("attention", initializer=tf.random_normal_initializer()):
                if attention_type == 'nn':
                    #attention by nn begin
                    att_w = tf.get_variable('att_w', [self.rnn_cell_dim * 2, self.attention_dim], tf.float32)
                    att_u = tf.get_variable('att_u', [self.rnn_cell_dim * 2 + self.feature_dim * 2, self.attention_dim],
                                            tf.float32)
                    att_b = tf.get_variable('att_b', [self.attention_dim], tf.float32)
                    att_v = tf.get_variable('att_v', [self.attention_dim, 1], tf.float32)

                    att_ht = tf.tensordot(outputs, att_w, axes=1)  # L, N , 10
                    att_h = tf.tensordot(state, att_u, axes=1)  # N, 10
                    e = att_ht + att_h + att_b  # L, N, 10
                    e = tf.transpose(e, perm=[1, 0, 2])  # N,L, 10
                    e = tf.nn.elu(e)
                    e = tf.tensordot(e, att_v, axes=[[2], [0]])
                    e = tf.reshape(e, shape=[self.batch_size, self.seq_length])  # N, L
                    att_value = tf.nn.softmax(e)
                    weighted_ht = tf.transpose(outputs, perm=[2, 1, 0]) * att_value
                    att_outputs = tf.transpose(tf.reduce_sum(weighted_ht, axis=2), perm=[1, 0])
                    #attention by nn end

                elif attention_type == 'bilinear':
                    #attention by bilinearity begin
                    self.bi_matrix = tf.get_variable('bi_matrix', [self.rnn_cell_dim * 2, self.rnn_cell_dim * 2 + self.feature_dim * 2], tf.float32)
                    e = tf.tensordot(outputs, self.bi_matrix, axes=[[2], [0]])
                    e = tf.transpose(e, perm=[1, 0, 2])
                    state = tf.reshape(state, shape=[self.batch_size, -1, 1])
                    e = tf.matmul(e,state)
                    e = tf.reshape(e, shape=[self.batch_size, self.seq_length])
                    att_value = tf.nn.softmax(e)
                    weighted_ht = tf.transpose(outputs, perm=[2, 1, 0]) * att_value
                    att_outputs = tf.transpose(tf.reduce_sum(weighted_ht, axis=2), perm=[1, 0])
                    #attention by bilinearity end

            #self.final_out = tf.concat([self.X_last], 1)  # N, 3*D        
            #self.final_out = tf.concat([att_outputs, self.X_last], 1)  # N, 3*D        
            self.final_out = tf.concat([att_outputs, self.X_last, matchtype_emb, cate_emb, trigger_emb, age_class_emb, baby_stage_emb, career_type_emb, self.Rxftrl_features], 1)  # N, 3*D        

            #####       Dense Classification    #####
            self.fc1 = tf.layers.dense(self.final_out, self.dense_dim, activation=tf.nn.relu, name='fc1', kernel_initializer = tf.random_normal_initializer())
            self.fc2 = tf.layers.dense(self.fc1, 2, activation=None, name='fc2')
            #self.fc2 = tf.layers.dense(self.final_out, 2, activation=None, name='fc2')
            pred_logit = tf.nn.softmax(self.fc2)
            return pred_logit

    def __load_data__(self):
        # pos,neg sample combine     by muming
        if self.is_debug == True:
            table = '../data/xftrlfeatures_0823_500.txt'#os.path.join('../train_data_01.csv')

        else:
            tables = self.table_name.split(',')
            if self.is_train:
                table = tables[0]
            else:
                table = tables[1]
            #selected_cols = 'user_id,cate_list,brand_list,item_list,time_list,action_list,show_single_keyword,pv_reach_time,click,qid,query_features,term_list,ner_list,l1_cateid1,l1_cateid2,l1_cateid3,ds,hh,rand_value'
            selected_cols = 'user_id,shop_list,cate_list,catelevel1_list,brand_list,item_list,item_features_list,time_list,action_list,item_name_hash_list,i2q_term_hash_list,show_single_keyword,pv_reach_time,click,ds,hh,qid,query_features,query_term_list,query_hash_list,l1_cateid1,l1_cateid2,l1_cateid3,rand_value,matchtype,cateid,triggerid,score_features,age_class,baby_stage,career_type,xftrl_features'

        filename_queue = tf.train.string_input_producer([table])
        print("reading data from", table)
        #record_defaults = [['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], [0.0], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-']]
        record_defaults = [['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], [0.0], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-'], ['-']]

        if self.is_debug == True:
            reader = tf.TextLineReader()
            key, value = reader.read(filename_queue)
            tmp = tf.decode_csv(value, record_defaults=record_defaults, field_delim='^')

        else:
            reader = tf.TableRecordReader(selected_cols=selected_cols, csv_delimiter='|')
            qkey, qvalue = reader.read(queue=filename_queue)
            tmp = tf.decode_csv(qvalue, record_defaults=record_defaults, field_delim='|')

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3*self.batch_size
        tmp = tf.train.shuffle_batch_join([tmp], batch_size=self.batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

        #tmp = tf.train.batch_join([tmp], batch_size=self.batch_size, capacity=capacity)

        Tuser_id, Tshop_list, Tcate_list, Tcatelevel1_list, Tbrand_list, Titem_list, Titem_features_list, Ttime_list, Taction_list, Titem_name_hash_list, Ti2q_term_hash_list, Tshow_single_keyword, Tpv_reach_time, Tclick, Tds, Thh, Tqid, Tquery_features, Tquery_term_list, Tquery_hash_list, Tl1_cateid1, Tl1_cateid2, Tl1_cateid3, Trand_value, Tmatchtype, Tcateid, Ttriggerid, Tscore_features, Tage_class, Tbaby_stage, Tcareer_type, Txftrl_features = tmp
       
        Rcate_list = tensor_from_file(Tcate_list, '-', self.seq_length, [self.seq_length, self.batch_size])
        Rcatelevel1_list = tensor_from_file(Tcatelevel1_list, '-', self.seq_length, [self.seq_length, self.batch_size])
        Rbrand_list = tensor_from_file(Tbrand_list, '-', self.seq_length, [self.seq_length, self.batch_size])
        Ritem_list = tensor_from_file(Titem_list, '-', self.seq_length, [self.seq_length, self.batch_size])
        Rtime_list = tensor_from_file(Ttime_list, 0.0, self.seq_length, [self.seq_length, self.batch_size])
        Raction_list = tensor_from_file(Taction_list, 0, self.seq_length, [self.seq_length, self.batch_size])
        Ritem_name_hash_list = tensor_from_file(Titem_name_hash_list, '-', self.seq_length*self.term_length, [self.seq_length, self.term_length, self.batch_size])
        Ri2q_term_hash_list = tensor_from_file(Ti2q_term_hash_list, '-', self.seq_length*self.term_length,  [self.seq_length, self.term_length, self.batch_size])
        Ritem_features_list = tensor_from_file(Titem_features_list, 0.0, self.seq_length*self.item_features_dim, [self.seq_length, self.item_features_dim, self.batch_size])
        Rquery_hash_list = tensor_from_file(Tquery_hash_list, '-', self.term_length, [self.term_length, self.batch_size])
        Rquery_features = tensor_from_file(Tquery_features, 0.0, self.query_features_dim, [self.query_features_dim, self.batch_size])
        Ruserid = tensor_from_file(Tuser_id, '-', 1, [1, self.batch_size])
        Rqid = tensor_from_file(Tqid, '-', 1, [1, self.batch_size])
        Rcateid1 = tensor_from_file(Tl1_cateid1, '-', 1, [self.batch_size])
        Rcateid2 = tensor_from_file(Tl1_cateid2, '-', 1, [self.batch_size])
        Rcateid3 = tensor_from_file(Tl1_cateid3, '-', 1, [self.batch_size])
        Rscore_features = tensor_from_file(Tscore_features, 0.0, 2, [2, self.batch_size])
        Rxftrl_features = tensor_from_file(Txftrl_features, 0.0, 14, [14, self.batch_size])
        Rmatchtype = tensor_from_file(Tmatchtype, '-', 1, [self.batch_size])
        Rcateid = tensor_from_file(Tcateid, '-', 1, [self.batch_size])
        Rtriggerid = tensor_from_file(Ttriggerid, '-', 1, [self.batch_size])
        Rage_class = tensor_from_file(Tage_class, '-', 1, [self.batch_size])
        Rbaby_stage = tensor_from_file(Tbaby_stage, '-', 1, [self.batch_size])
        Rcareer_type = tensor_from_file(Tcareer_type, '-', 1, [self.batch_size])

        Rcate_list = tf.string_to_number(tf.substr(Rcate_list,0,7), tf.int32)
        Rcatelevel1_list = tf.string_to_number(tf.substr(Rcatelevel1_list,0,7), tf.int32)
        Rbrand_list = tf.string_to_number(tf.substr(Rbrand_list,0,7), tf.int32)
        Ritem_list = tf.string_to_number(tf.substr(Ritem_list,0,7), tf.int32)
        Ruserid = tf.string_to_number(tf.substr(Ruserid,0,7), tf.int32)
        Rqid = tf.string_to_number(tf.substr(Rqid,0,7), tf.int32)
        Rcateid1 = tf.string_to_number(tf.substr(Rcateid1,0,7), tf.int32)
        Rcateid2 = tf.string_to_number(tf.substr(Rcateid2,0,7), tf.int32)
        Rcateid3 = tf.string_to_number(tf.substr(Rcateid3,0,7), tf.int32)
        Ritem_name_hash_list = tf.string_to_number(tf.substr(Ritem_name_hash_list,0,7), tf.int32)
        Ri2q_term_hash_list = tf.string_to_number(tf.substr(Ri2q_term_hash_list,0,7), tf.int32)
        Rquery_hash_list = tf.string_to_number(tf.substr(Rquery_hash_list,0,7), tf.int32)
        Rmatchtype = tf.string_to_number(tf.substr(Rmatchtype,0,7), tf.int32)
        Rcateid = tf.string_to_number(tf.substr(Rcateid,0,7), tf.int32)
        Rtriggerid = tf.string_to_number(tf.substr(Rtriggerid,0,7), tf.int32)
        Rage_class = tf.string_to_number(tf.substr(Rage_class,0,7), tf.int32)
        Rbaby_stage = tf.string_to_number(tf.substr(Rbaby_stage,0,7), tf.int32)
        Rcareer_type = tf.string_to_number(tf.substr(Rcareer_type,0,7), tf.int32)

        click_block = tf.stack([Tclick], axis=0)
        label_value = tf.reshape(click_block, shape=[-1])
        label_oh = tf.one_hot(indices=tf.cast(label_value, tf.int32), depth=2, dtype=tf.float32)

        print('Rcate_list.shape, Rquery_features.shape, Rqid.shape, label_oh.shape:'), Rcate_list.shape, Rquery_features.shape, Rqid.shape, label_oh.shape

        return Ruserid, Rcate_list, Rcatelevel1_list, Rbrand_list, Ritem_list, Rtime_list, Raction_list, Ritem_name_hash_list, Ri2q_term_hash_list, Ritem_features_list, Rquery_hash_list, Rquery_features, Rqid, Rcateid1, Rcateid2, Rcateid3, Rscore_features, Rxftrl_features, Rmatchtype, Rcateid, Rtriggerid, Rage_class, Rbaby_stage, Rcareer_type, label_value, label_oh

    def __data__(self):
        print('load data')
        return self.__load_data__()
