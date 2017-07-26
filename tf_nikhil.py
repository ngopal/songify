import tensorflow as tf
import utils
import sys, os

# load graph
with open("models/vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print("graph loaded from disk")

graph = tf.get_default_graph()

# analyze images in directory specified by sys.argv[1]
loaded_images = [(i, utils.load_image(sys.argv[1]+os.sep+i)) for i in os.listdir(sys.argv[1])]
cumulative_description_words = []

for imloc, img in loaded_images:
  with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print("variables initialized")

    batch = img.reshape((1, 224, 224, 3))
    assert batch.shape == (1, 224, 224, 3)

    feed_dict = { images: batch }

    prob_tensor = graph.get_tensor_by_name("import/prob:0")
    prob = sess.run(prob_tensor, feed_dict=feed_dict)

  print(imloc)
  utils.print_prob(prob[0])
  cumulative_description_words.append(  utils.get_top(prob[0])  )

print(cumulative_description_words)







