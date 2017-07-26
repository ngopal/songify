from VideoProcessing import VideoProc
from Models import DocModel

class Vid2Song:
    def __init__(self):
        self.VP = VideoProc()
        self.Model = DocModel()

if __name__ == "__main__":
    App = Vid2Song()
    #img_keywords = App.VP.analyzeImagesInDir("./testvideos/imagedragonframes")
    img_keywords = ['n09472597 volcano', 'n04404412 television, television system', 'n06359193 web site, website, internet site, site', 'n04404412 television, television system', 'n04404412 television, television system', 'n04404412 television, television system', 'n03782006 monitor', 'n03250847 drumstick', 'n03832673 notebook, notebook computer', 'n04286575 spotlight, spot', 'n04296562 stage', 'n01910747 jellyfish', 'n04404412 television, television system', 'n03782006 monitor', 'n03729826 matchstick']
    App.Model.nullModel(img_keywords, "4SMubSJhL8oHG1RNa6RGkQ")


    # 4SMubSJhL8oHG1RNa6RGkQ ## Sound of Seattle