# Most of script lifted from stackoverflow
# https://stackoverflow.com/questions/56766639/how-to-convert-ckpt-to-pb

import os
import tensorflow as tf
import argparse

def convert_graph_to_graphviz(graph):
    """Converts a TensorFlow graph to a Graphviz dot file.

    Args:
        graph: A TensorFlow graph.

    Returns:
        A string containing the Graphviz dot file.
    """
    dot = ['digraph graphname {']
    for node in graph.as_graph_def().node:
        dot.append('  "{}"'.format(node.name))
        if node.op == 'Placeholder':
            dot.append(' [label="{}"]'.format(node.name))
        dot.append(';')
        for input_node in node.input:
            dot.append('  "{}" -> "{}";'.format(input_node, node.name))
    dot.append('}')
    return dot.join(os.linesep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("tf_ckpt_to")
    parser.add_argument("--checkpoint", help="The checkpoint prefix path for example: models/model.ckpt-49491", type=str, required=True)
    parser.add_argument("--output_format", help="The output format for example: pb", type=str, required=True)
    parser.add_argument("-o/--output_dir", help="The output directory to put the .pb file.")
    args = parser.parse_args()
    parser.add_mutually_exclusive_group

    trained_checkpoint_prefix = args.checkpoint
    export_dir = args.output_dir

    # Validate user input for output_dir
    assert(args.output_format in ["pb", "saved_model", "dot"], "Output format must be one of pb, saved_model, or dot")

    # Load the input graph into a TensorFlow graph
    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        if args.output_format == "pb" or args.output_format == "saved_model":

            # Export checkpoint to SavedModel
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                                 strip_default_attrs=True)
            builder.save() 
        elif args.output_format == "dot":
            with open(os.join(export_dir, "graph.dot"), "w") as f:
                f.write(convert_graph_to_graphviz(graph))
        else:
            raise ValueError("Output format must be one of pb, saved_model, or dot")