import pickle
class FileUtil:
    @staticmethod
    def savemodel(model,filename):
        try:
            pickle.dump(model,open(filename,"wb"))
            return True
        except:
            print("An exception Occurred")
            return False
    @staticmethod
    def loadmodel(filename):
        try:
            model = pickle.load(open(filename,"rb"))
            return model
        except:
            print("An exception Occurred")
            return None
