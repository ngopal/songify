from VideoProcessing import VideoProc
from Models import DocModel

class Vid2Song:
    def __init__(self):
        self.VP = VideoProc()
        self.Model = DocModel()

if __name__ == "__main__":
    App = Vid2Song()
    img_keywords = App.VP.analyzeImagesInDir("./testvideos/imagedragonframes")
    App.Model.nullModel(img_keywords)