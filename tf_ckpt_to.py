# Most of script lifted from stackoverflow
# https://stackoverflow.com/questions/56766639/how-to-convert-ckpt-to-pb

import os
import tensorflow as tf
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("tf_ckpt_to_pb")
    parser.add_argument("--checkpoint", help="The checkpoint prefix path for example: models/model.ckpt-49491", type=str, required=True)
    parser.add_argument("-o/--output_dir", help="The output directory to put the .pb file.")
    args = parser.parse_args()
    parser.add_mutually_exclusive_group

    trained_checkpoint_prefix = args.checkpoint
    export_dir = args.output_dir

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                             strip_default_attrs=True)
        builder.save() 
