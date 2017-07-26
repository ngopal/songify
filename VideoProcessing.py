import tensorflow as tf
import utils
import sys, os
from subprocess import call



class VideoProc:
    def __init__(self):
        # load graph
        with open("NNmodels/vgg16.tfmodel", mode='rb') as f:
            self.fileContent = f.read()

        self.graph_def = tf.GraphDef()
        self.graph_def.ParseFromString(self.fileContent)

        self.images = tf.placeholder("float", [None, 224, 224, 3])

        tf.import_graph_def(self.graph_def, input_map={ "images": self.images })
        print("graph loaded from disk")

        self.graph = tf.get_default_graph()

    def analyzeImagesInDir(self, dir):
        # analyze images in directory specified by sys.argv[1]
        loaded_images = [(i, utils.load_image(dir+os.sep+i)) for i in os.listdir(dir)]
        cumulative_description_words = []

        for imloc, img in loaded_images:
            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                print("variables initialized")

                batch = img.reshape((1, 224, 224, 3))
                assert batch.shape == (1, 224, 224, 3)

                feed_dict = { self.images: batch }

                prob_tensor = self.graph.get_tensor_by_name("import/prob:0")
                prob = sess.run(prob_tensor, feed_dict=feed_dict)

            #print(imloc)
            #utils.print_prob(prob[0])
            cumulative_description_words.append(  utils.get_top(prob[0])  )

        return cumulative_description_words

    def downSampleVideo(self,videoloc,outdir,frames):
        command = 'ffmpeg -i "'+videoloc+'" -vf fps=1/'+str(frames)+' -f image2 "'+outdir+os.sep+'video-frame%03d.png"'
        call([command], shell=True)
        return


if __name__ == "__main__":
    VP = VideoProc()
    # VP.downSampleVideo("./testvideos/imaginedragons.mp4", "./testvideos/imagedragonframes", 15)
    print(VP.analyzeImagesInDir("./testvideos/imagedragonframes"))








