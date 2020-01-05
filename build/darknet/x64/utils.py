import numpy as numpy
from darknet import *
import os
import imageio
import glob
import sys
from shutil import copyfile
from skimage import io, draw
import imageio
import cv2


def get_npz_list(path_given):
    if not os.path.exists(path_given):
        raise ValueError("Invalid path given `"+os.path.abspath(path_given)+"`")
    
    namecounter = 0
    pathlist = []
    npz_list_return = []
    os.chdir(path_given)
    dir_list_1 = sorted(glob.glob('*/'))
    if len(dir_list_1) == 0:
        # no underlying directories
        file_list = glob.glob('*.npz')
        for file1 in file_list:
            filepath = os.path.normpath(path_given + '/' + file1)
            #data = np.load(filepath)
            #data1 = data['arr_0']
            #data1 = np.flipud(data1)
            #data1 = np.fliplr(data1)
            #os.chdir(sys.path[0])
            #imageio.imsave('./temp/npz2video/' + str(namecounter) + '.jpg', data1)
            #copyfile('./temp/npz2video/' + str(namecounter) + '.npz', dst)
            npz_list_return.append(filepath)
            namecounter = namecounter+1
            print(namecounter)
    else:
        # another layer of underlying directories
        for dir1 in dir_list_1:
            new_file_path = path_given + '/' + dir1
            new_file_path = os.path.normpath(new_file_path)
            os.chdir(new_file_path)
            file_list = sorted(glob.glob('*.npz'))
            for file1 in file_list:
                filepath = os.path.normpath(new_file_path + '/' + file1)
                #data = np.load(filepath)
                #data1 = data['arr_0']
                #data1 = np.flipud(data1)
                #data1 = np.fliplr(data1)
                #os.chdir(sys.path[0])
                #imageio.imsave('./temp/npz2video/' + str(namecounter) + '.jpg', data1)
                #namecounter = namecounter+1
                #print(namecounter)
                npz_list_return.append(filepath)
    return npz_list_return


def npz2video(dirpath, thresh, configPath, weightPath, metaPath):
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if not os.path.exists(dirpath):
        raise ValueError("Invalid directory path `"+os.path.abspath(imagePath)+"`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)	# if is used cv2.imread(image)
    video_name = 'video.avi'
    counter = 0
    pathlist = get_npz_list(dirpath)
    image = io.imread(pathlist[0])
    os.chdir(sys.path[0])
    imageio.imwrite('./temp/npz2video/singlepic.jpg', image)
    image = io.imread("./temp/npz2video/singlepic.jpg")
    height, width = image.shape
    outpath = "/home/leon/software/darknetdata"
    framerate = 4
    os.chdir(outpath)
    video = cv2.VideoWriter(video_name, 0, framerate, (width,height))
    os.chdir(sys.path[0])
    
    for imagePath in pathlist:
        detections = detect(netMain, metaMain, imagePath, thresh)
        try:
            image = io.imread(imagePath)
            image = np.flipud(image)
            image = np.fliplr(image)
            imageio.imwrite('./temp/npz2video/singlepic.jpg', image)
            image = io.imread("./temp/npz2video/singlepic.jpg")
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)    

            
            imcaption = []
            for detection in detections:
                label = str(detection[0])
                confidence = detection[1]
                pstring = label+": "+str(np.rint(100 * confidence))+"%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2]/2)
                yCoord = int(bounds[1] - bounds[3]/2)
                boundingBox = [
                    [xCoord, yCoord],
                    [xCoord, yCoord + yExtent],
                    [xCoord + xEntent, yCoord + yExtent],
                    [xCoord + xEntent, yCoord]
                ]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox], shape= shape)
                rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox], shape= shape)
                rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox], shape= shape)
                #image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha= 0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha= 0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha= 0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha= 0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha= 0.8)
                imageio.imwrite('./temp/npz2video/singlepic.jpg', image)
                image = io.imread("./temp/npz2video/singlepic.jpg")
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Unable to show image: "+str(e))
        video.write(image)
        print("Converted image number: ", counter, "\n")
        counter = counter+1
    cv2.destroyAllWindows()
    video.release()

