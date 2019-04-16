#-*- coding:utf-8 -*-
import os, sys, traceback

import tensorflow as tf
from tensorflow.python.client import timeline

import model, utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.app.flags.DEFINE_string("tables","./train.txt", "tables_info")
tf.app.flags.DEFINE_string("tests", "", "test info")
tf.app.flags.DEFINE_string("checkpointDir", "../log/", "oss info")
tf.app.flags.DEFINE_string("exp", "default", "experiment name")
tf.app.flags.DEFINE_string("m1", "1", "Modification 1")
tf.app.flags.DEFINE_string("m2", "1", "Modification 2")
tf.app.flags.DEFINE_string("m3", "1", "Modification 3")
FLAGS = tf.app.flags.FLAGS

debug = False

if FLAGS.tests == '':
    FLAGS.tests = FLAGS.tables

assert (FLAGS.checkpointDir != "" and FLAGS.checkpointDir[-1] is '/'), \
    "[Assert Error] checkpointDir must be specified and must end with /"

m = model.Seq2Point(table_name=FLAGS.tables, checkpoint_dir=FLAGS.checkpointDir,
                    epochs=100, batch_size=500, feature_dim=10, query_features_dim=7, 
                    m1=FLAGS.m1, m2=FLAGS.m2, m3=FLAGS.m3, is_debug = debug)
m_test = model.Seq2Point(table_name=FLAGS.tests, checkpoint_dir=FLAGS.checkpointDir,
                         epochs=100, batch_size=500, feature_dim=10, query_features_dim=7, 
                         m1=FLAGS.m1, m2=FLAGS.m2, m3=FLAGS.m3, is_train=False, is_debug = debug)
m.build_input()
m_test.build_input()

with tf.variable_scope("top") as scope:
    m.build_graph()
    scope.reuse_variables()
    m_test.build_graph()

m.build_summary()
m_test.build_summary('test')

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
run_options = tf.RunOptions() if 1 else tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
saver = tf.train.Saver(max_to_keep=None)
print('================')
print(m.optimizer)
if m.optimizer is None:
    print('None')

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    print('Start Session')
    sess.run(init)
    print('1')
    sess.run(local_init)
    print('2')
    writer = tf.summary.FileWriter( os.path.join(FLAGS.checkpointDir, 'tensorboard/'+FLAGS.exp+'/train/'), sess.graph)
    print('3')
    twriter = tf.summary.FileWriter( os.path.join(FLAGS.checkpointDir, 'tensorboard/'+FLAGS.exp+'/test/'), sess.graph)
    print('4')
    coord = tf.train.Coordinator()
    print('5')
    threads = tf.train.start_queue_runners(coord=coord)
    print('6')
    step = 0
    try:
        print('Start Train')
        while not coord.should_stop():
            with utils.log_time() as log:
                for i in range(2):
                    Xone, X1, X2, Rcate_list, final_out, fc2, loss_v, X, X_last, label_oh, pred_logit, positive_score, pred_binary, label_value, summary, _, step = sess.run([m.Xone, m.X1, m.X2, m.Rcate_list, m.final_out, m.fc2, m.loss, m.X, m.X_last, m.label_oh, m.pred_logit, m.positive_score, m.pred_binary, m.label_value, m.summary_op, m.optimizer, m.global_step], options=run_options, run_metadata=run_metadata)
                log.write(u'iteration: %d'%step)
                print ('Xone:', Xone)
                print ('X1:', X1)
                print ('X2:', X2)
                print ('final_out:', final_out)
                #print ('fc1:', fc1)
                print ('fc2:', fc2)
                #print ('X:', X)
                print ('X_last:', X_last)
                print ('label_oh:', label_oh)
                print ('pred_logit:', pred_logit)
                print ('positive_score:', positive_score)
                print ('pred_binary:', pred_binary)
                print ('label_value:', label_value)
                print ('loss:', loss_v)
                writer.add_summary(summary, step)

                summary = sess.run(m_test.summary_op)
                twriter.add_summary(summary, step)

            if step>1 and step%4000 == 0:
                ckp_path = os.path.join(FLAGS.checkpointDir, 'model_saved', str(step)+'_model.ckpt')
                path = saver.save(sess, ckp_path, step)
                print ('model saved at {}'.format(path))
    except:
        traceback.print_exc(file=sys.stdout)
    finally:
        writer.close()
        coord.request_stop()
        coord.join(threads)
