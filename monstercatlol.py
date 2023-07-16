import multiprocessing 
# import the necessary packages
#pip install cython
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
#from scipy.interpolate.UnivariateSpline import *
import tornado.ioloop
import tornado.web
import numpy as np
import argparse
import imutils
import array
from pydub.utils import get_array_type
from time import time
import cv2
from pydub import AudioSegment
from multiprocessing import Pool
from multiprocessing import cpu_count
import subprocess
import matplotlib.pyplot as plt
import sys
import os
import time 
import itertools
import operator ## only needed if want to play with operators
from imutils.video import FPS, WebcamVideoStream
#import subprocess
import matplotlib
import multiprocessing as mp
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from itertools import accumulate
import time
import collections
FPS = 30
M_PI = 3.1415926535897932385
from scipy.signal import savgol_filter
from scipy.fftpack import dct
import js2py
import pygame
from scipy.signal import savgol_filter
#averageTransform = js2py.eval_js('function averageTransform(array) {var values = [];var length = array.length; for (var i = 0; i < length; i++) { var value = 0;if (i == 0) {value = array[i];} else if (i == length - 1) {value = (array[i - 1] + array[i]) / 2;} else {var prevValue = array[i - 1];var curValue = array[i];var nextValue = array[i + 1];if (curValue >= prevValue && curValue >= nextValue) {  value = curValue;} else {value = (curValue + Math.max(nextValue, prevValue)) / 2;}}value = Math.min(value + 1, 150);values[i] = value;}var newValues = [];for (var i = 0; i < length; i++) {var value = 0;if (i == 0) {    value = values[i];} else if (i == length - 1) {    value = (values[i - 1] + values[i]) / 2;} else {    var prevValue = values[i - 1];    var curValue = values[i];    var nextValue = values[i + 1];if (curValue >= prevValue && curValue >= nextValue) {value = curValue;} else {value = ((curValue / 2) + (Math.max(nextValue, prevValue) / 3) + (Math.min(nextValue, prevValue) / 6));}}value = Math.min(value + 1, 150);newValues[i] = value;}return newValues;}')
#tailTransform = js2py.eval_js('function tailTransform(array,headMargin=7,) {var values = []; for (var i = 0; i < array.length; i++) { var value = array[i]; value *= tailMarginSlope * Math.pow(array.length - i, marginDecay) + minMarginWeight; }values[i] = value;}return values;}')

sumItAll = 5
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    
    def get(self):
        self.write(str(sumItAll))

def Convert(lst):
    res_dct = {i: lst[i] for i in range(0, len(lst))}
    return res_dct

class MainHandlerSpectrum(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'application/json')

    
    def get(self):
        self.write(Convert(spectrum))


class MainHandlerKick(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    
    def get(self):
        if isKick == True:
            self.write("1")
        else:
            self.write("0")

class MainHandlerHighhat(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    
    def get(self):
        if isHighhat == True:
            self.write("1")
        else:
            self.write("0")

class MainHandlerSnare(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    
    def get(self):
        if isSnare == True:
            self.write("1")
        else:
            self.write("0")

class FPS:
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0

def getFreq(x):
    return int(21.675 * math.exp(0.109*x))

freqIndex = [  26,48,73,93,115,138,162,185,207,231,254,276,298,323,346,370,392,414,436,459,483,507,529,552,575,598,621,644,669,714,828,920,1057,1173,1334,1472,1655,1840,2046,2253,2483,2735,3012,3287,3609,3930,4275,4665,5056,5493,5929,6412,6917,7446,7998,8618,9261,9928,10617,11352,11996,12937,13718,14408]
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--maxbars", default=64,   help="How many max bars? (64 bars for a monstercat effect!)")
ap.add_argument("-i", "--initial", default=0.00,   help="Level Initial")
ap.add_argument("-ti", "--trebleinitial", default=0.00,   help="Treble Level Initial")
ap.add_argument("-hi", "--high", default=474,   help="High frequency bin")
ap.add_argument("-l", "--low", default=4,   help="Low frequency bin")
ap.add_argument("-hi1", "--high1", default=10000,   help="High frequency cutoff")
ap.add_argument("-l1", "--low1", default=10,   help="Low frequency  cutoff")
ap.add_argument("-g", "--initialtreble", default=0.00,   help="Treble Level Initial.")
ap.add_argument("-v", "--volume", default=0.045, help="volume.")
ap.add_argument("-ga", "--gain", default=0.05,   help="Treble volume.")
ap.add_argument("-q", "--levelMax", default=0.67,   help="Smoothing Decay")
ap.add_argument("-1", "--level", default=0.67,   help="Smoothing Attack")
ap.add_argument("-t", "--maxtreblebars", default=43,   help="Offset treble bars? (42 for monstercat effect!)")
ap.add_argument("-th", "--maxhightreblebars", default=47,   help="Offset high treble bars? (42 for monstercat effect!)")
ap.add_argument("-2", "--smlevel", default=12,   help="Smoothing Level (Monstercat Style)") # was 1.5 or 3
ap.add_argument("-4", "--sm2level", default=3,   help="Smoothing Level 2 (Monstercat Style)")
ap.add_argument("-3", "--treblesmlevel", default=3,   help="Treble Smoothing Level (Monstercat Style)") # maybe the same with maxbars?
#treblehighsmlevel
ap.add_argument("-5", "--treblehighsmlevel", default=3,   help="High Treble Smoothing Level (Monstercat Style)") # maybe the same with maxbars?
ap.add_argument("-s1", "--passes", default=3,   help="Smoothing Passes")
ap.add_argument("-mam", "--measureaudiomultiplier", default=40,   help="....")
ap.add_argument("-sens", "--sensitivity", default=35,   help="....")
ap.add_argument("-sp", "--spacing", default=16,   help="...")
ap.add_argument("-si", "--size", default=13,   help="...")
ap.add_argument("-fi", "--mcfilter", default=2.0,   help="...")
ap.add_argument("-wf", "--wavefilter", default=0.0,   help="...")
ap.add_argument("-sa", "--samples", default=44100,   help="...") # 48000
ap.add_argument("-fft", "--fftsize", default=2000,   help="...") # 14!
ap.add_argument("-buff", "--buffsize", default=2000,   help="...") # 12!
ap.add_argument("-int", "--integral", default=85,   help="...")
ap.add_argument("-tint", "--trebleintegral", default=85,   help="...")
ap.add_argument("-gra", "--gravity", default=1500,   help="...")
ap.add_argument("-eq", "--eqbalance", default=0.64,   help="...")
ap.add_argument("-ls", "--logscale", default=1.64,   help="...")
ap.add_argument("-lm", "--limit", default=1500,   help="height limit")
args = vars(ap.parse_args())
maxVolume = float(args["volume"])
maxBars = (int(args["maxbars"]))
isHighhat = False
isKick = False
isSnare = False
isChecking = False
# Plot values in opencv program
class Plotter:
	def __init__(self, plot_width, plot_height):
            self.width = plot_width
            self.height = plot_height
            self.color = (255, 0 ,0)
            self.val = []
            self.fig = Figure(figsize=(self.width,self.height), dpi=100)
            self.plot_canvas = FigureCanvasAgg(self.fig)

	# Update new values in plot
	def plot(self, val, label = "plot"):
		self.val.append(int(val))
		while len(self.val) > self.width:
			self.val.pop(0)

		self.show_plot(label)

    # Show plot using opencv imshow
	def show_plot(self, label):
		cv2.line(self.plot_canvas, (i, int(self.height/2) - self.val[i]), (i+1, int(self.height/2) - self.val[i+1]), self.color, 5)

		cv2.imshow(label, self.plot_canvas)
		cv2.waitKey(10)

duration = 0.01

fig = plt.figure()


# initialize the video stream and allow the camera
# sensor to warmup
print("[INFO] warming up camera...")
#filename = args["input"]

#data,sample_rate1 = librosa.load(filename, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')
#time.sleep(2.
# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
#fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)
zeros = None
#video = VideoFileClip(args["input"])
# loop over frames from the video stream
clips = 0.01
fps = 0.00
tick = 0
frameCounter = 1
#timeBegin = time()
texts = 0
txt_clips = 0
prev_frame_time = 0
new_frame_time = 0
result = 0
print("loading analysis...");
import time
#audioReact = {isBass: False, isKick: False, isSnare: False}
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = VideoWriter(args["output"],frameSize=(1280,720))

#out.open()
timeout = 0
fpslist = []
FullFPSEst = 0
#import sys, math, wave, numpy, pygame

last_unique_frame = 1
fps_video = 30
total_fps_counter = 0
actual_fps = 0
fps_to_display = 0
identical_frame = 1
fps_countdown = 1000
frame_time = 1000/fps_video
plot_x = 0
plot_y = 0
seconds = 1
isGettingCap = False
started = time.time ()
capDelta = 0
tickindex=0;
ticksum=0;
seconds = 1
samples = 0
framesRendered = 1
framerate = 0
scale = 1
avgFrameRate = 1
last_from_time = time.time()
fpsListing = []
#f = []
fdiff = []
fc_list = []
differ = 0.0
rfr = 0.0
timeMeasurements = []
timeMeasurements.insert(0, time.time())
thenny = time.time()
def shift(key, array):
    return array[-key:]+array[:-key]
#pool = multiprocessing.Pool()
#pool.map(gen_plot_frame, range(0,8))
#pool.map(_get_difference, range(0,8))
#pool.map(_get_frame_difference, range(0,8))
import scipy
import scipy.fftpack as fftpk
total_fps_counter = 0
#MAX_FPS = int(args['maxfps'])
avg_fps = 0.0
last_tick = cv2.getTickCount()
fps = []
import math
last_frame = None
last_frame_set = False
#fpsLists = []
getActualFps = 0.0
fpsCountDown = 1000
identical_frame = 1
last_unique_frame = time.time()
fpsListing.insert(0,1.0)
fpsListing.insert(-1,1.0)
fpsListing.insert(1,1.0)
#f.insert(0,cv2.getTickCount()-last_tick)
#f.insert(-1,cv2.getTickCount()-last_tick)
fps = FPS()
low = 20
high = 20
volume = 0
#gain = (int(args["gain"]))
amplitude = [1.0] * (maxBars+16) # 64 bars
amplitudes = [1.0] * (maxBars+16) # 64 bars
spectrum = [1.0] * (maxBars+16) # 64 bars
realSpectrum = [1.0] * (maxBars+16) # 64 bars
bars = [1.0] * (maxBars+16) # 64 bars
bars1 = [1.0] * (maxBars+16) # 64 bars
barsCutoff = [1.0] * (maxBars+16)
now = time.time()
SMOOTHING_FACTOR = float(args['initial'])
#fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
#writer = cv2.VideoWriter(args["output"], fourcc, 60.0, (1280, 720))
from pydub import AudioSegment 
from pydub.playback import play
import pyaudio
from numpy import sqrt 
#import pygame
from scipy.signal import butter, sosfilt, sosfreqz

from scipy.signal import butter, lfilter, filtfilt
def frange(start=0, stop=1, jump=0.1):
    nsteps = int((stop-start)/jump)
    dy = stop-start
    # f(i) goes from start to stop as i goes from 0 to nsteps
    return [start + float(i)*dy/nsteps for i in range(nsteps)]
import random
from enum import Enum
silenced = True
barsSilenced = True
def butter_highpass(cutoff, nyq_freq, order=0):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='highpass', analog = False)
    return b, a

def butter_highpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_highpass(cutoff_freq, nyq_freq, order=order)
    y = lfilter(b, a, data)
    return y
def butter_lowpass(cutoff, nyq_freq, order=0):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog = False)
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = lfilter(b, a, data)
    return y
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog = False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=1)
    y = lfilter(b, a, data)
    return y

def smooth1(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth(y):
    box = np.ones(1)/1
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

from numpy import array, diff, where, split
def findPeak(magnitude_values, noise_level=2000):
    
    splitter = 0
    # zero out low values in the magnitude array to remove noise (if any)
    magnitude_values = numpy.asarray(magnitude_values)        
    low_values_indices = magnitude_values < noise_level  # Where values are low
    magnitude_values[low_values_indices] = 0  # All low values will be zero out
    
    indices = []
    
    flag_start_looking = False
    
    both_ends_indices = []
    
    length = len(magnitude_values)
    for i in range(length):
        if magnitude_values[i] != splitter:
            if not flag_start_looking:
                flag_start_looking = True
                both_ends_indices = [0, 0]
                both_ends_indices[0] = i
        else:
            if flag_start_looking:
                flag_start_looking = False
                both_ends_indices[1] = i
                # add both_ends_indices in to indices
                indices.append(both_ends_indices)
                
    return indices
def convert_2d(x, y, z, horizon):
    d = 1 - (z/horizon)
    return x*d, y*d
def extractFrequency(indices, freq_threshold=2):
    
    extracted_freqs = []
    
    for index in indices:
        freqs_range = indices[index[0]: index[1]]
        avg_freq = round(numpy.average(freqs_range))
        
        if avg_freq not in extracted_freqs:
            extracted_freqs.append(avg_freq)

    # group extracted frequency by nearby=freq_threshold (tolerate gaps=freq_threshold)
    group_similar_values = split(extracted_freqs, where(diff(extracted_freqs) > freq_threshold)[0]+1 )
    
    # calculate the average of similar value
    extracted_freqs = []
    for group in group_similar_values:
        extracted_freqs.append(round(numpy.average(group)))
    
    #print("freq_components", extracted_freqs)
    return extracted_freqs

def draw_rect_alpha(surface, color, rect, size=8, alpha=10):
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    shape_surf.set_alpha(alpha)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect(),size)
    surface.blit(shape_surf, rect)
from pygame.locals import *
import numpy

flags = pygame.DOUBLEBUF
DEFAULT_IMAGE_SIZE = (200, 200)
pygame.init()
win = pygame.display.set_mode([1280, 720],flags,16)
screen = win
modes = pygame.display.list_modes(8)
if not modes:
    print('16-bit not supported')
else:
    print('Found Resolution:', modes[0])
    pygame.display.set_mode(modes[0], FULLSCREEN, 16)
clock = pygame.time.Clock ()     # Each chunk will consist of 1024 samples
sample_format = pyaudio.paInt16      # 16 bits per sample
channels = 2     # Number of audio channels
fs = 44100
# Record at 44100 samples per second
time_in_seconds = 1
isDone = True
size = width, height = (1280, 720)
smoothing_multiplier = [0] * (maxBars*16)
smoothing_multiplier_one = [0] * (maxBars*16)
empty_surface = pygame.Surface(size)
empty_surface_1 = pygame.Surface(size)
offset = (maxBars/4)

def draw_stars_cos(screen, speed, maxStars, Width, Height):
    for i in range(int(maxStars)):
        for x in range(Width):
            formX = (math.cos(x-(maxStars+i)*(Width+speed))/1000)
            formY = (math.cos(x-(maxStars+i)*(Height+speed))/1000)
            pygame.draw.circle(screen, (255,255,255), (formX,formY), 2)
from scipy.ndimage.interpolation import shift
def savitskyGolaySmooth(array, smoothingPoints=3, smoothingPasses=5):
    lastArray = array
    for pass1 in range(smoothingPasses):
        sidePoints = math.floor(smoothingPoints / 2)
        cn = 1 / (2 * sidePoints + 1)
        newArr = [0] * len(array)
        for i in range(sidePoints):
            newArr[i] = lastArray[i]
            newArr[len(lastArray) - i - 1] = lastArray[len(lastArray) - i - 1]
        for i in range((len(lastArray) - sidePoints)):
            sum1 = 0
            for n in range(-sidePoints, sidePoints):
                sum1 += cn * lastArray[i + n] + n
            newArr[i] = sum1
        lastArray = newArr
    return newArr
marginDecay = 1.6
headMargin = 7
tailMargin = 1
minMarginWeight = 0.7
headMarginSlope = (1-minMarginWeight) / pow(headMargin, marginDecay)
tailMarginSlope = 0
def tailTransform(array):
    array = np.asarray(array)
    values = [0] * len(array)
    for i in range(len(array)):
        value = array[i]
        if (i < headMargin):
            value *= headMarginSlope * pow(i + 1, marginDecay) + minMarginWeight
        elif (maxBars - i <= tailMargin):
            value *= tailMarginSlope * pow(int(args["size"]) - i, marginDecay) + minMarginWeight
        values[i] = value
    return values
spectrumMaxExponent = 6
spectrumExponentScale = 2
spectrumMinExponent = 3

def howManyNewTails(array, tails=2):
    newArr = [0]*(len(array)+tails)
    for i in range(len(array)):
        for j in range(tails):
            newArr[i+j] = (4+(newArr[i])/(1+j))
    return newArr

def averageTransform(array):
    array = np.asarray(array)
    values = [0] * (len(array)*2)
    length = (len(array)-1)

    for i in range(length):
        value = 0
        if (i == 0):
            value = array[i]
        elif (i == length - 1):
            value = (array[i - 1] + array[i]) / 2
        else:
            prevValue = array[i - 1]
            curValue = array[i]
            nextValue = array[i + 1]
            if (curValue >= prevValue and curValue >= nextValue):
              value = curValue
            else:
              value = (curValue + max(nextValue, prevValue)) / 2
        #value = min(value + 1, int(720/4))

        values[i] = value

    newValues = [0] * (len(array)*2)
    for i in range(length):
        value = 0
        if (i == 0):
            value = array[i]
        elif (i == length - 1):
            value = (array[i - 1] + array[i]) / 2
        else:
            prevValue = array[i - 1]
            curValue = array[i]
            nextValue = array[i + 1]
            if (curValue >= prevValue and curValue >= nextValue):
              value = curValue
            else:
              value = ((curValue / 2) + (max(nextValue, prevValue) / 3) + (min(nextValue, prevValue) / 6))
        #value = min(value + 1, int(720/4))

        newValues[i] = value
    
    return newValues

spectrumScale = 2.5
spectrumStart = int(args["low"])
spectrumEnd = int(args["high"])
def transformToVisualBins(array):
    array = np.asarray(array)
    newArray = [0] * (len(array)+8)
    for i in range(maxBars):
        #print(i)
        bin = pow(i / maxBars, spectrumScale) * (spectrumEnd - spectrumStart) + spectrumStart
        #print(bin)
        newArray[i] = array[math.floor(bin) + spectrumStart] * (bin % 1) + array[math.floor(bin + 1) + spectrumStart] * (1 - (bin % 1))
    return newArray

def exponentialTransform(array):
    array = np.asarray(array)
    newArr = [0] * len(array)
    for i in range(len(array)):
        exp = spectrumMaxExponent + (spectrumMinExponent - spectrumMaxExponent) * (i/len(array))
        newArr[i] = max(pow(array[i] / 150, exp) * 150, 1)
    return newArr
def normalizeAmplitude(array):
    values = [0] * len(array)
    for i in range(len(array)):
        values[i] = (array[i]/255) * volume
        values[i] = min(values[i], int(args["limit"]))
    return values

trebleBars = int(args["maxtreblebars"])
def calcTreble(array):
    newArray = []
    for i in range(len(array)):
        d = (i + (array[i]-array[i-1]))
        newArray.append(d)

def expTransform(array):
    array = np.asarray(array)
    resistance = int(args["sm2level"])
    newArr = [0] * len(array)
    for i in range(len(array)):
        sum = 0
        divisor = 0
        for j in range(len(array)):
            dist = abs(i - j)
            weight =  1 / pow(2, dist)
            if weight == 1:
                weight = resistance
            sum += int(array[j]) * weight
            divisor += weight
        newArr[i] = sum / divisor
    return newArr

attackD = [0.0] * (maxBars+8)
def Attack(array):
    array = np.asarray(array)
    for i in range(1,maxBars):
        #print(attackD[i])
        attackD[i] = int(abs((attackD[i] * float(args["levelMax"])) + ((abs(((array[i])))) * (1-float(args["levelMax"])))))
    return attackD
decayD = [0.0] * (maxBars+8)
def Decay(array):
    
    for i in range(1,maxBars):
        #print(attackD[i])
        decayD[i] = int(abs((attackD[i] * float(args["level"])) + ((abs(((array[i])))) * (1-float(args["level"])))))
    return decayD
flast = [0] * (maxBars+8)
fall = [0] * (maxBars+8)
#f = [0] * (maxBars+8)
fpeak = [0] * (maxBars+8)
#def getPeaks(array):
#    array = np.asarray(array)
#    return 
def FalloffFilt(f, gravity):
    #array = np.asarray(array)
    gravity = (gravity * 50.0) / 100
    g = gravity * ((150-1) / 2160)
    for i in range(1,maxBars):
        if (f[i] < flast[i]):
            if(fall[i] == 0):
                fall[i] = time.time()

            time_diff = (time.time() - fall[i]) / 16.0
            flast[i] = fpeak[i] - (g * time_diff * time_diff)
            f[i] = min(0, flast[i])
            fall[i] += 1
        else:
            fpeak[i] = f[i]
            fall[i] = 0
            
    return f
def Filter(array, waves, monstercat):
    monstercat = monstercat * 1.5
    array = np.asarray(array)
    if(waves > 0):
        for z in range(len(array)):
            array[z] = array[z] / 1.25
            for m_y in range(z - 1,0):
                de = z - m_y
                array[m_y] = max(array[z] - pow(de, 2), array[m_y])
            for m_y in range(len(array),z+1):
                de = m_y - z
                array[m_y] = max(array[z] - pow(de, 2), array[m_y])
    elif (monstercat > 0):
         for z in range(len(array)):
            for m_y in range(z - 1,0):
                de = z - m_y
                array[m_y] = max(array[z] - pow(monstercat, 2), array[m_y])
            for m_y in range(len(array),z+1):
                de = m_y - z
                array[m_y] = max(array[z] - pow(monstercat, 2), array[m_y])
    return array

logScale = float(args['logscale'])
eqBalance = float(args['eqbalance'])
cuttoffs = [0] * (maxBars + 1)
def CutoffFreq(array,lower,upper):
    cut_off_frequency = [0] * (maxBars + 8)
    eq = [0] * (maxBars + 8)
    fc = [0] * (maxBars + 8)
    lcf = [0] * (maxBars + 8)
    hcf = [0] * (maxBars + 8)
    fre = [0] * (maxBars + 8)
    k = [0] * (maxBars + 8)
    relative_cut_off = [0] * (maxBars + 8)
    array = np.asarray(array)
    frequency_constant = math.log(upper-lower)/math.log(pow(maxBars, logScale))
    #bar_distribution_coefficient = frequency_constant * (-1)
    for n in range(maxBars+1):
        #bar_distribution_coefficient += (n + 1) / (maxBars + 1) * frequency_constant
        fc[n] = pow(pow(n, (logScale-1.0)*(n+1.0)/(maxBars+1.0)),frequency_constant)+lower
        #if n > 0:
        #    if(cut_off_frequency[n-1] >= cut_off_frequency[n] and cut_off_frequency[n - 1] > 100):
        #        cut_off_frequency[n] = cut_off_frequency[n-1] + (cut_off_frequency[n-1] - cut_off_frequency[n-2])
        fre[n] = fc[n] / (fs / 2.0)
        lcf[n] = math.floor(fre[n] * (fs/2.0))
        if n != 0:
            hcf[n-1] = lcf[n]
        #if maxBars > 1:
    for n in range(maxBars+1):
        k[n] = pow(fc[n], eqBalance)
        k[n] += array[n]
        #print(array)
    return k

def OddOneOut(array):
    #f = [0] * (maxBars + 8)
    array = np.asarray(array)
    f = array
    for i in range(1,maxBars-1):
        f[i] = (f[i+1] + f[i-1])/2
    for i in range((maxBars-1),1):
        sum1 = (f[i+1] + f[i-1])/2
        if(sum>f[i]):
            f[i] = sum1
    return f
#array2 = [0] * (maxBars+8)
fmem = [0] * (maxBars+8)
def Integral(array,gravity):
    #f = FalloffFilt(array,gravity)
    f = array
    integral = int(args["integral"]) / 100
    tintegral = int(args["trebleintegral"]) / 100
    array = np.asarray(array)
    for i in range(1,maxBars):
        #f[i] = fmem[i] * integral + f[i]
        if i>=(trebleBars):
            f[i] = fmem[i] * tintegral + f[i]
        elif i<(trebleBars):
            f[i] = fmem[i] * integral + f[i]
        fmem[i] = f[i]
        diff = 1500 - f[i]
        if(diff < 0):
            diff = 0
        div = 1.0 / (diff + 1)
        fmem[i] = abs(fmem[i] * (1 - div / 20))
    return fmem


def resize(array, new_size, new_value=0):
    """Resize to biggest or lesser size."""
    element_size = len(array) #Quantity of new elements equals to quantity of first element
    if new_size > len(array):
        new_size = new_size - 1
        while len(array)<=new_size:
            n = tuple(new_value for i in range(element_size))
            array.append(n)
    else:
        array = array[:new_size]
    return array

def monstercatify(array):
    array = np.asarray(array)
    arrayLower = [0] * len(array)
    newArray = [0] * len(array)
    mcatify = averageTransform(array)
    for i in range(len(array)):
        if mcatify[i] >= 10:
            arrayLower[i] = 10
        else:
            arrayLower[i] = mcatify[i]
        #newArray[i] = ((arrayLower[i])+((array[i]/15) * 10))
        if i>=(trebleBars):
            arrayLower[i] = 0
        #    newArray[i] = (((array[i]/20) * 15))
        newArray[i] = ((arrayLower[i])+((array[i]/15) * 10))
    return newArray


def monstercat_filter(array):
    ic = 0
    newArr2 = [0] * (len(array)+8)
    mcat = array
    #newArr2 = [0] * (len(array)+8)
    #smootht = smooth1(array, 3)
    #SmoothT_2 = averageTransform(smooth1(array,abs((maxBars-5))))
    smootht = savitskyGolaySmooth(array, 3, 1)
    #Smmoth = smooth1(SmoothT, maxBars)
    #SmoothT_1 = savitskyGolaySmooth(array, abs(((maxBars-trebleBars)-16)), abs(int(args["maxhightreblebars"])))
    #SmoothT_1 = savitskyGolaySmooth(array, (int(args["maxhightreblebars"])), abs((int(args["maxhightreblebars"]-1))))
    #SmoothT_2 = smooth1(array,3)
    sumTreble = 0
    for i in range(1,maxBars):
        #newArr2[i-1] = ((array[i-1]+array[i]+array[i+1]/float(args["smlevel"]))/8)*2
        newArr2[i] = ((((smootht[i-1]/3)+(mcat[i-1]))+((smootht[i]/3)+(mcat[i]))+((smootht[i+1]/3)+(mcat[i+1]))/float(args["smlevel"])))
        #newArr2[i] = mcat[i]
        #newArr2[i+1] = ((array[i-1]+array[i]+array[i+1]/float(args["smlevel"]))/8)
        #newArr2[i+1] = (array[i-1]+array[i]+array[i+1]/float(args["smlevel"]))
		
#	if i >= (trebleBars):
        if(i>=(trebleBars-5) and i < (trebleBars)):
           newArr2[i] = ((((smootht[i-1]))+((smootht[i]))+((smootht[i+1]))/float(args["smlevel"])))
        if(i>=(trebleBars-4) and i < (trebleBars-1)):
           newArr2[i] = ((((smootht[i-1]))+((smootht[i]))+((smootht[i+1]))/float(args["smlevel"])))
        if i > (trebleBars):
            passes = 3
            p = 1
            #prevV_sm = SmoothT_2[i-1] # (trebleBars-8)
            #currV_sm = SmoothT_2[i]
            #nextV_sm = SmoothT_2[i+1]
            #prevV = OddOneOut(savitskyGolaySmooth(array, 3, 1))[i-1]
            #currV = OddOneOut(savitskyGolaySmooth(array, 3, 1))[i]
            #nextV = OddOneOut(savitskyGolaySmooth(array, 3, 1))[i+1]
            prevV = ((smootht[i-1])) * ((i*1.10)/maxBars)
            currV = ((smootht[i])) * ((i*1.10)/maxBars)
            nextV = ((smootht[i+1])) * ((i*1.10)/maxBars)
            #print(((i*2)/maxBars))
                #print("yes")
            #avgArr = averageTrform(array)
            #newArr2[i] = ((savitskyGolaySmooth(newArr2,passes,p)[i-1])+(savitskyGolaySmooth(array,passes,p)[i])+(savitskyGolaySmooth(array,passes,p)[i+1])/float(args["smlevel"])) # ((maxBars-trebleBars)-16)
            #newArr2[i] = ((prevV+currV+nextV/float(args["smlevel"])))
            #newArr2[i] = (((prevV+currV+nextV/8)) + ((array[i-1]+array[i]+array[i+1]/8)))
            #newArr2[i] = (((prevV + currV + nextV ))) # / 64 - ((array[i-1] + array[i] + array[i+1] / 8))
            newArr2[i] = ((prevV + currV + nextV / float(args["smlevel"])))
            if newArr2[i] <= 0.00000000000000000009:
                newArr2[i] = 0
#            if(i>=(maxBars-2)):
#                newArr2[i] = (((prevV + currV + nextV / 12 )))
#            else:
#                newArr2[i] = (((prevV + currV + nextV / 8 )))

        #sumTreble /= 2
        #newArr2[trebleBars] /= 3
        #newArr2[trebleBars] /= 2

   # prevV = (((smootht[trebleBars-1])))
   # currV = (((smootht[trebleBars])))
    #newArr2[trebleBars+1] -= eqBalance
    newArr2[trebleBars+1] -= (newArr2[(trebleBars+1)-1]/3)
   #newArr2[trebleBars] -= newArr2[trebleBars-2]
        #if i >= (int(args["maxhightreblebars"])):
        #    newArr2[i] += ((smootht[i-1]+smootht[i]+smootht[i+1]/float(args["treblehighsmlevel"])))
            #newArr2[i] /= 16
            
    #if newArr2[i] <= 0.00000000000000000009:
    #   print(i)
    #s   newArr2[i] = 0 # normalize
        #if newArr2[i]>0:
        #newArr2[i-2] = (newArr2[i]+2)
        #   newArr2[i+1] = (newArr2[i])
        #elif ic == 3:
        #    ic = 0
#        if newArr2[i]>0:
#            newArr2[i] = (newArr2[i]*8)
#            newArr2[i+1] = (newArr2[i]*4)
        #newArr2[i+1] = (newArr2[i]+(255-volume))
        #newArr2[i] = (newArr2[i])
        #newArr2[i+1] = (newArr2[i+1])
        #newArr2[i+2] = (newArr2[i+2]-2)
        #ic += 1
    #newArr3 = savitskyGolaySmooth(newArr2, 3, 1)
    return newArr2

def monstercat_smoothing(array):
    smoothing_w1 = [0] * len(array)
    smoothing_w = [0] * len(array)
    for i in range(1,len(array)):
        #print(pow(float(args["sm2level"]), i))
        smoothing_w1[i] = pow(float(args["sm2level"]), i)

    for i in range(1,len(array)):
        for j in range(1,len(array)):
            smoothing_w[i] = smoothing_w[i] / (1+smoothing_w1[i - j])
    return array

def getTransformedSpectrum(array):
    #newArr = butter_bandpass_filter(array,float(args["low1"]),float(args["high1"]),fs,1)
    
    newArr = CutoffFreq(array,float(args["low1"]),float(args["high1"]))
    newArr = normalizeAmplitude(newArr)
    #newArr = transformToVisualBins(array)
    #newArr = monstercat_smoothing(array)
    #newArr = averageTransform(newArr)
   # newArr = averageTransform(newArr)
    newArr = tailTransform(newArr)
    #newArr = expTransform(newArr)
    #newArr = CutoffFreq(newArr,float(args["low"]),float(args["high"]))
    #if itr >= s:
    #newArr = averageTransform(newArr)
    #newArr = tailTransform(newArr)
    #newArr = averageTransform(newArr)
    ##newArr = smooth(newArr, int(args['smoothq']))
    #newArr = tailTransform(newArr)
    #newArr = tailTransform(newArr)
    #newArr = tailTransform(newArr)
    #newArr = butter_bandpass_filter(newArr,1,8000,fs,3)
    #newArr = tailTransform(newArr)
    #newArr = np.tile(newArr, maxBars)
    #newArr = howManyNewTails(newArr, 4)
    #newArr = averageTransform(newArr)
    #newArr = savitskyGolaySmooth(newArr,3,1)
    #newArr = tailTransform(newArr)
    #print(newArr)
    #newArr = OddOneOut(newArr)
    #newArr = OddOneOut(newArr)
   # newArr = monstercat_filter(newArr)

    #newArr = Integral(newArr,int(args['gravity']))
    #newArr = averageTransform(newArr)
    #newArr = monstercatify(newArr)
    #newArr = expTransform(newArr)
    #newArr = monstercatify(newArr)
    #newArr = monstercat_filter(newArr)
    #newArr = tailTransform(newArr)
    
    newArr = Filter(newArr, float(args['wavefilter']), float(args['mcfilter']))
    #newArr = monstercatify(newArr)
    newArr = Integral(newArr,int(args['gravity']))
    newArr = monstercat_filter(newArr)
    #newArr = averageTransform(newArr)
    newArr = monstercat_smoothing(newArr)
    newArr = expTransform(newArr)
    newArr = OddOneOut(newArr)
    newArr = monstercatify(newArr)
    #newArr = averageTransform(newArr)
    #newArr = Attack(newArr)
    #newArr = Decay(newArr)
    
    #newArr = expTransform(newArr)
    #newArr = tailTransform(newArr)
    #newArr = Filter(newArr, float(args['wavefilter']), float(args['mcfilter']))
    #newArr = trebleCat(newArr)
    #newArr = savitskyGolaySmooth(newArr, float(args["smlevel"]), 1)
    #newArr = expTransform(newArr)
    #newArr = averageTransform(newArr
    #newArr = exponentialTransform(newArr)
    #newArr = smooth1(newArr,1) 
    #newArr = monstercatify(newArr)
	
    return newArr

import threading


def lerp(a, b, t):
    return(a + (b - a) * t)
def colorLerp(c1, c2, t):
    return((c1[0] + (c2[0] - c1[0]) * t, 
            c1[1] + (c2[1] - c1[1]) * t,
            c1[2] + (c2[2] - c1[2]) * t))

class Particle:
    def __init__(self, pos, color, opacity, size, speedX, speedY, lifespan):
        self.pos = pos
        self.color = color
        self.opacity = opacity
        self.size = size
        self.speedX = speedX
        self.speedY = speedY
        self.lifespan = lifespan
        self.deathClock = 0
    def render(self):
        pygame.draw.circle(screen, self.color, [int(self.pos[0]), int(self.pos[1])], int(self.size))

class OriginType(Enum):
    Point = 0
    Box = 1

class ParticleSystem:
    def __init__(self, origin, frequence, atATime, color, lifespan, xspread, gravity, randomLife, size, origintype, originSize):
        self.origin = origin
        self.particles = []
        self.frequence = frequence
        self.color = color
        self.lifespan = lifespan
        self.xspread = xspread
        self.gravity = gravity
        self.randomLife = randomLife
        self.size = size
        self.origintype = origintype
        self.originSize = originSize
        self.atATime = atATime
        self.timer = 0
    def update(self):
        self.timer += 1;
        if (self.timer >= self.frequence):
            self.timer = 0;
            for k in range(self.atATime):
                randx = random.random()
                self.particles.append(Particle(self.origin[:] if self.origintype == OriginType.Point else [self.origin[0] + self.originSize[0] * randx, self.origin[1] + self.originSize[1] * random.random()], 
                                               self.color[:],
                                               1, 
                                               self.size, 
                                               (random.random()*self.xspread-(self.xspread/2)) if self.origintype == OriginType.Point else (random.random() * self.xspread * (randx*2 - 1)), 
                                               0, 
                                               self.lifespan + (random.random()*self.lifespan*self.randomLife)-(self.lifespan/2)))
        
        for i in range(len(self.particles)):
            self.particles[i].deathClock += 1
        self.particles = [part for part in self.particles if part.deathClock < part.lifespan]
            
        for i in range(len(self.particles)):
            particleTime = self.particles[i].deathClock/self.particles[i].lifespan
            if isinstance(self.size, list):
                self.particles[i].size = lerp(self.size[0], self.size[1], particleTime)
            if isinstance(self.color, list):
                self.particles[i].color = colorLerp(self.color[0], self.color[1], particleTime)
            self.particles[i].speedY += self.gravity;
            self.particles[i].pos[0] += self.particles[i].speedX
            self.particles[i].pos[1] += self.particles[i].speedY
    def render(self):
        for part in self.particles:
            part.render()

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))
class WebServer(tornado.web.Application):

    def __init__(self):
        handlers = [ (r"/", MainHandler), (r"/spectrum", MainHandlerSpectrum), (r"/reactKick", MainHandlerKick), (r"/reactHHat", MainHandlerHighhat), (r"/reactSnare", MainHandlerSnare), ]
        settings = {'debug': True}
        super().__init__(handlers, **settings)

    def run(self, port=8886):
        self.listen(port)
        tornado.ioloop.IOLoop.instance().start()
import asyncio
ws = WebServer()
import particlepy
def start_server():
    asyncio.set_event_loop(asyncio.new_event_loop())
    #ws.listen(8888)
    ws.run()

def volumeUpStarted():
    #clock.tick(20)
    isChecking = True
    global barsSilenced
    for b in range(maxBars):
        #screen.fill(GREEN)
        #clock.tick(60)
        bar = [int(1280/12), int(720/2), int((b+(b*int(args["spacing"])))), 4]
        pygame.draw.rect(empty_surface,(255,255,255),bar,0)
        img_with_flip = pygame.transform.flip(empty_surface, False, True)
        screen.blit(img_with_flip, (0, 0))
        pygame.display.flip()
    barsSilenced = False
    print("starting...")
    d = threading.Thread(target=volumeUpEnded)
    d.daemon = True
    d.start()
    isChecking = False
def volumeUpEnded():
    #global barsSilenced
    global volume
    global maxVolume
    for i in frange(0,maxVolume,0.01):
        #clock.tick(60)
        clock.tick(35)
        print(i)
        volume = i    
def volumeDownStarted():
    #clock.tick(40)
    global isChecking
    global volume
    global maxVolume
    isChecking = True
    for i in frange(0,maxVolume,0.01):
        clock.tick(35)
        volume = maxVolume-i
        print(maxVolume-i)
    d = threading.Thread(target=volumeDownEnded)
    d.daemon = True
    d.start()
def volumeDownEnded():
    global barsSilenced
    barsSilenced = True
    print("ending...")
    for b in range(maxBars):
        #clock.tick(60)
        bar = [int(1280/12), int(720/2), int(((maxBars+(maxBars*int(args["spacing"])))-(b+(b*int(args["spacing"]))))), 4]
        pygame.draw.rect(empty_surface,(255,255,255),bar,0)
        img_with_flip = pygame.transform.flip(empty_surface, False, True)
        screen.blit(img_with_flip, (0, 0))
        pygame.display.flip()
    isChecking = False
import audioop
rms = None
rmsTreble = None
def callback(in_data, frame_count, time_info, status):
    global rms
    rms = in_data
    return in_data, pyaudio.paContinue

def callback1(in_data, frame_count, time_info, status):
    global rmsTreble
    rmsTreble = in_data
    return in_data, pyaudio.paContinue
highs = int(args["high"])
lows = int(args["low"])
if __name__ == "__main__":
    print('-----Now Recording-----')
   #procs = cpu_count()
   #procIDs = list(range(0,procs)) 
   #pool = Pool(processes=procs)
   #pool = mp.Pool(mp.cpu_count())
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                channels = 2,
                rate = fs,
                frames_per_buffer = (3000+maxBars),
                input = True,
                stream_callback=callback)
    streamTreble = p.open(format=sample_format,
                channels = 2,
                rate = fs,
                frames_per_buffer = 2000,
                input = True,
                stream_callback=callback1)
    stream.start_stream()
    streamTreble.start_stream()
    #data = sound._data
   #initFPS = 30
    ind = 32
    #pygame.transform.flip(empty_surface, True, True)
    t0 = 20
    t1 = 20
    #app = make_app()
    #server = tornado.httpserver.HTTPServer(app)
    #server.listen(8888)
    #server.start(0)
    #tornado.ioloop.IOLoop.current().start()
    x = threading.Thread(target=start_server)
    x.daemon = True
    x.start()
    #x.join()
    #partsys = ParticleSystem([350, 250], 1, 2, [(255, 255, 0), (255, 0, 0)], 70, 1, -0.15, 0.2, [6, 0], OriginType.Box, [15, 15])
    #smoke = ParticleSystem([358, 250], 1, 2, [(60, 60, 60), (0, 0, 0)], 150, 2, -0.15, 0.2, [10, 0], OriginType.Point, [15, 15])
    #particle_system = particlepy.particle.ParticleSystem()
    old_time = time.time()
    delta_time = 0
    while stream.is_active() and streamTreble.is_active():
        if rms != None and rmsTreble != None:
            #print(rms)
            #clock.tick(30)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        maxVolume += 0.01
                        if silenced == False:
                            volume += 0.01
                    if event.key == pygame.K_DOWN:
                        maxVolume -= 0.01
                        if silenced == False:
                            volume -= 0.01
                    if event.key == pygame.K_LEFT:
                        if silenced == True:
                            d = threading.Thread(target=volumeUpStarted)
                            d.daemon = True
                            d.start()
                            silenced = False
                        elif silenced == False:
                            d = threading.Thread(target=volumeDownStarted)
                            d.daemon = True
                            d.start()
                            silenced = True                    
            now = time.time()
            delta_time = now - old_time
            old_time = now
            #particle_system.update(delta_time=delta_time)
            #print(len(particle_system.particles))
            #clock.tick(MAX_FPS)
            #frames.append(data)
            #frame = np.zeros((720, 1280, 3), dtype = "uint8")*indx
            #frame[:] = (0, 255, 0)
            #partsys.update()
            #smoke.update()
            screen.fill(GREEN)
            empty_surface.fill((0,255,0))
            empty_surface_1.fill(GREEN)
            #screen.set_alpha(None)
            #chunks = int(chunk)
            #data = stream.read(2000)
            waveform = np.frombuffer(rmsTreble, dtype=np.int16)
            waveform1 = np.frombuffer(rms, dtype=np.int16)
            #waveform = transformToVisualBins(waveform)
            window = scipy.signal.windows.triang(maxBars)
            filtereddata = CutoffFreq(transformToVisualBins(numpy.fft.fft(waveform1,fs))[lows:highs], int(args["low1"]), int(args["high1"]))
            #filtereddata = transformToVisualBins(filtereddata
            filtereddata2 = numpy.fft.rfft(filtereddata,n=(maxBars+10))
            fft_complex = numpy.fft.ifft(filtereddata2,n=maxBars)
            
            filtereddata_t_mid = numpy.fft.fft(waveform)[1250:2500+maxBars]
            #filtereddata_t_mid = transformToVisualBins(filtereddata_t_mid)
            fft_complex_t_mid = numpy.fft.ifft(filtereddata_t_mid[:maxBars],n=maxBars)[:maxBars]
            filtereddata_t_mid_2 = numpy.fft.fft(waveform)[900:2000+maxBars]
            #butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_mid_2,n=maxBars)[:maxBars], 5000, 10000, fs, 1)
            #filtereddata_t_mid_2 = transformToVisualBins(filtereddata_t_mid_2)
            fft_complex_t_mid_2 = numpy.fft.ifft(filtereddata_t_mid_2[:maxBars],n=maxBars)[:maxBars]
            filtereddata_t_kick = numpy.fft.fft(waveform)[50:100+maxBars]
            #filtereddata_t_kick = transformToVisualBins(filtereddata_t_kick)
            fft_complex_t_kick = butter_bandpass_filter(filtereddata_t_kick, 50, 100, fs, maxBars)
            #butter_bandpass_filter(newArr,1,8000,fs,3)
            filtereddata_t_snare = numpy.fft.fft(waveform)[120:250+maxBars]
            #filtereddata_t_snare = transformToVisualBins(filtereddata_t_snare)
            #fft_complex_t_snare = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_snare)[:maxBars], 120, 250, fs, 1)
            fft_complex_t_snare = numpy.fft.ifft(filtereddata_t_snare)[:maxBars]
            filtereddata_t_snare_2 = numpy.fft.fft(waveform)[300:600+maxBars]
            filtereddata_t_hihats_iso = numpy.fft.fft(waveform)[300:3000+maxBars]
            fft_complex_t_hihats_iso = butter_bandpass_filter(filtereddata_t_hihats_iso, 300, 3000, fs, maxBars)
            filtereddata_t_snares_iso = numpy.fft.fft(waveform)[120:250+maxBars]
            fft_complex_t_snares_iso = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_snares_iso)[:maxBars], 120, 250, fs, maxBars)
            filtereddata_t_claps_iso = numpy.fft.fft(waveform)[6000:20000+maxBars]
            fft_complex_t_claps_iso = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_snares_iso)[:maxBars], 6000, 20000, fs, maxBars)
            #filtereddata_t_snare = transformToVisualBins(filtereddata_t_snare)
            #fft_complex_t_snare_2 = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_snare_2)[:maxBars], 300, 600, fs, 1)
            fft_complex_t_snare_2 = numpy.fft.ifft(filtereddata_t_snare_2)[:maxBars]
            filtereddata_t_snare_thump = numpy.fft.fft(waveform)[200:300+maxBars]
            filtereddata_t_snare_thump_end_2 = numpy.fft.fft(waveform)[2000:3000+maxBars]
            fft_complex_t_snare_thump_end_2 = butter_bandpass_filter(filtereddata_t_snare_thump_end_2, 2000, 3000, fs, maxBars)
            #filtereddata_t_snare = transformToVisualBins(filtereddata_t_snare)
            #fft_complex_t_snare_thump = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_snare_thump)[:maxBars], 200, 300, fs, 1)
            fft_complex_t_snare_thump = numpy.fft.ifft(filtereddata_t_snare_thump)[:maxBars]
            filtereddata_t_sizzle = numpy.fft.fft(waveform)[10000:10000+maxBars]
            #filtereddata_t_snare = transformToVisualBins(filtereddata_t_snare)
            #fft_complex_t_sizzle = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_sizzle,n=maxBars)[:maxBars], 6000, 7500, fs, 1)
            fft_complex_t_sizzle = numpy.fft.ifft(filtereddata_t_sizzle,n=maxBars)[:maxBars]
            #print(fft_complex_t_sizzle)
            filtereddata_t_voice = numpy.fft.fft(waveform)[80:255]
            fft_complex_t_voice = numpy.fft.ifft(filtereddata_t_voice[:maxBars])[:maxBars]
            filtereddata_t_crash = numpy.fft.fft(waveform)[4000:6000+maxBars]
            fft_complex_t_crash = butter_bandpass_filter(filtereddata_t_snare_thump_end_2, 4000, 6000, fs, 1)
            #print(fft_complex_t_crash)
            filtereddata_t_bass = numpy.fft.fft(waveform)[20:60+maxBars]
            fft_complex_t_bass = butter_bandpass_filter(numpy.fft.ifft(filtereddata_t_bass)[:maxBars], 20, 60, fs, maxBars)

            filtereddata_t_treble = numpy.fft.fft(waveform)[getFreq(trebleBars):getFreq(maxBars)+maxBars]
            fft_complex_t_treble = numpy.fft.ifft(filtereddata_t_treble,n=maxBars)[:maxBars]

            filtereddata_t_b = numpy.fft.fft(waveform)[6000:20000]
            fft_complex_t_b = numpy.fft.ifft(filtereddata_t_b,n=maxBars)[:maxBars]

            #print(fft_complex_t_treble)
            #print(fft_complex_t_voice)
            #print(fft_complex_t_snare)
            pos = []
            barObj = []
            #s = pygame.Surface((1000,750))
            #s.fill((0,255,0))
            #s.set_alpha(128)
            #frame.fill(255)
            #print(str(indx)+" of "+str(len(sound))+"\r\n")
            #screen.fill([0,255,0])
            ind1 = int((maxBars)/2)
            sum1 = 0
            bias = 0
            bass_cut_off = 150
            treble_cut_off = 2500
            mD = 0
            mD_1 = 1
            ease = 1
            if (ease == 2):
                ease = 1
            treble_i = 1
            for i in range(int((maxBars))):
                if(i>=0 and (i%2)):
                    #print(i)
                    if(i<(trebleBars-5)): #  - 5
                        wave_bass = wave_bass_t_mid = ((i+10)/(maxBars-3)*np.abs(fft_complex)) / 10
                        #if (ease == 1
                        valueMag = wave_bass[i]
                        #if SMOOTHING_FACTOR != -1:
                        #bars[i] = abs(((int(window[i]+valueMag))))
                        bars[i] = int(abs((bars[i] * float(args["initial"])) + ((abs(((int(valueMag))))) * (1-float(args["initial"])))))
                        #else:
                        #bars[i] = int(abs(((window[i]+valueMag))))
                    if(i>=(trebleBars-5) and i < (trebleBars)):
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_mid = smooth1(((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid)), maxBars)*(32+treble_i)
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_snare_thump2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_thump_end_2))*10
                        wave_bass_t_crash = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        wave_bass_t_kick = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_kick))*4
                        wave_bass_t_hihats = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_hihats_iso))*10
                        #fft_complex_t_hihats_iso
                        #wave_bass_t_crash = shift(wave_bass_t_crash,-5)
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*10
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(10)
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        sumAllsHighEnds = 0
                        for i2 in range(maxBars):
                            sumAllsHighEnds += (wave_bass_t_snare_thump[i2])
                        sumAllsKicks = 0
                        for i2 in range(maxBars):
                            sumAllsKicks += (wave_bass_t_kick[i2])
                        kick = 0
                        #if((wave_bass_t_kick[i])>1000):
                          #print((wave_bass_t_kick[i]))
                        kick = (wave_bass_t_kick[i]*24)
                        #print(wave_bass_t_claps[i])
                        kickSl = int(abs((bars[i] * float(0.80)) + ((abs(((int(kick))))) * (1-float(0.80)))))
                        valueMag = ((((wave_bass[i]*2)+(wave_bass_t_treble[i]*4)+(wave_bass_t_claps[i]*2))-(wave_bass_t_mid[(maxBars-4)])+wave_bass_t_treble[maxBars-1]-(wave_bass_t_treble[(maxBars-1)]+wave_bass_t_hihats[i]-((wave_bass_t_mid[i]+wave_bass_t_mid_2[i]))))-kickSl)
                        if(valueMag <= 0):
                          valueMag = 0
     #print(valueMag)
                        #if SMOOTHING_FACTOR != -1:
                        #bars[i] = int(abs((bars[i] * float(args["initialtreble"])) + ((abs(((int(window[i]+valueMag))))) * (1-float(args["initialtreble"])))))
                        #else:
                        #if (ease == 1):
                        valueMag *= 2
                        bars[i] = int(abs((bars[i] * float(0.90)) + ((abs(((int(valueMag/3))))) * (1-float(0.90)))))
                        treble_i += 1
                    if(i>=(trebleBars-4) and i < (trebleBars-1)):
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_mid = smooth1(((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid)), maxBars)*(32+treble_i)
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_snare_thump2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_thump_end_2))*10
                        wave_bass_t_crash = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_crash))*4
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*10
                        wave_bass_t_snare_thump2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_thump_end_2))*10
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        wave_bass_t_kick = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_kick))*4
                        wave_bass_t_hihats = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_hihats_iso))*10
			#fft_complex_t_claps_iso
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(10)
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        sumAllsHighEnds = 0
                        for i2 in range(maxBars):
                            sumAllsHighEnds += (wave_bass_t_snare[i2]+wave_bass_t_snare_2[i2]+wave_bass_t_snare_thump[i2])
                        sumAllsKicks = 0
                        for i2 in range(maxBars):
                            sumAllsKicks += (wave_bass_t_kick[i2])
                        #print(sumAllsKicks)
                        #print(wave_bass_t_claps[i])
                        kick = 0
                        #if((wave_bass_t_kick[i])>1000):
                          #print((wave_bass_t_kick[i]))
                        kick = (wave_bass_t_kick[i]*24)
                        #print(wave_bass_t_claps[i])
                        kickSl = int(abs((bars[i] * float(0.80)) + ((abs(((int(kick))))) * (1-float(0.80)))))
                        valueMag = ((((wave_bass[i]*12)+(wave_bass_t_treble[i]*4)+(wave_bass_t_claps[i]*2))-(wave_bass_t_mid[(maxBars-4)])+wave_bass_t_treble[maxBars-1]-(wave_bass_t_treble[(maxBars-1)]+wave_bass_t_hihats[i]-((wave_bass_t_mid[i]+wave_bass_t_mid_2[i]))))-kickSl)
                        if(valueMag <= 0):
                          valueMag = 0
			    #print(valueMa)g
                        #if SMOOTHING_FACTOR != -1:
                        #bars[i] = int(abs((bars[i] * float(0.90)) + ((abs(((int(window[i]+valueMag))))) * (1-float(0.90)))))
                        #else:
                        #if (ease == 1):
                        valueMag *= 4
                        bars[i] = int(abs((bars[i] * float(0.90)) + ((abs(((int(valueMag)))))) * (1-float(0.90))))
                        treble_i += 1
                        #bars[i] = int(abs(((window[i]+valueMag))))
                    elif(i>=(trebleBars) and i<(maxBars-14)):
                        #wave_bass_t = smooth1(getTransf/ormedSpectrum(((i+100) / maxBars * np.abs(fft_complex_t))),4)
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_sizzle))
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #wave_bass_t_mid = smooth1(wave_bass_t_mid,maxBars)*4
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        #print(wave_bass_t_claps[i])
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_voice[i]+wave_bass[i]+wave_bass_t_treble[i]+wave_voice[i]+wave_bass_t_claps[i])))) * (1-float(args["trebleinitial"]))))
                        #if (ease == 1):
                        #bars[i] = abs(window[i]+wave_bass_t_mid[i]+wave_bass_t_snare[i])
                        #bars[i] = abs((bars[i] * SMOOTHING_FACTOR) + ((abs(((bars1[i])))) * (1-SMOOTHING_FACTOR)))
                        treble_i += 1
                    elif(i>=(maxBars-14) and i<(maxBars-9)):
                        #wave_bass_t = smooth1(getTransf/ormedSpectrum(((i+100) / maxBars * np.abs(fft_complex_t))),4)
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*4
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #wave_bass_t_mid = smooth1(wave_bass_t_mid,maxBars)*4
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_voice[i]+wave_bass_t_treble[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_claps[i])))) * (1-float(args["trebleinitial"]))))
                        #if (ease == 1):
                        #bars[i] = abs(window[i]+wave_bass_t_mid[i]+wave_bass_t_snare[i])
                        #bars[i] = abs((bars[i] * SMOOTHING_FACTOR) + ((abs(((bars1[i])))) * (1-SMOOTHING_FACTOR)))
                        treble_i += 1
                    elif(i>=(maxBars-9) and i<(maxBars-7)):
                        #wave_bass_t = smooth1(getTransf/ormedSpectrum(((i+100) / maxBars * np.abs(fft_complex_t))),4)
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*4
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_bass_t_treble[i]+wave_bass_t_treble[i]+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_claps[i])))) * (1-float(args["trebleinitial"]))))
                        #if (ease == 1):
                        #bars[i] = abs(window[i]+wave_bass_t_mid[i]+wave_bass_t_snare[i])
                        #bars[i] = abs((bars[i] * SMOOTHING_FACTOR) + ((abs(((bars1[i])))) * (1-SMOOTHING_FACTOR)))
                        treble_i += 1
                    elif(i==(maxBars-7)):
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #wave_bass_t_crash = shift(wave_bass_t_crash,-5)
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*4
                        calcTrebleVal = int(((maxBars)-(maxBars-(maxBars+(maxBars)))/maxBars))
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        #print(calcTrebleVal/4)
                        #wave_bass_t_treble = shift(wave_bass_t_trele,-2)
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        #wave_bass_t_crash[i] /= 2
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_voice[i]+wave_bass_t_treble[i]+wave_bass_t_treble[i]+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_claps[i])))) * (1-float(args["trebleinitial"]))))
                        treble_i += 1
                    elif(i>=(maxBars-8) and i<(maxBars-5)):
                        #wave_bass_t = smooth1(getTransf/ormedSpectrum(((i+100) / maxBars * np.abs(fft_complex_t))),4)
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #wave_bass_t_crash = shift(wave_bass_t_crash,-5)
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*4
                        calcTrebleVal = int(((maxBars)-(maxBars-(maxBars+(maxBars)))/maxBars))
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        #print(calcTrebleVal/8)/
                        #wave_bass_t_treble = shift(wave_bass_t_trele,-2)
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        #wave_bass_t_crash[i] /= 2
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_bass_t_treble[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_mid_2[i]+wave_bass_t_claps[i])))) * (1-float(args["trebleinitial"]))))
                        treble_i += 1
                    elif(i>=(maxBars-4) and i<(maxBars-2)):
                        #wave_bass_t = smooth1(getTransf/ormedSpectrum(((i+100) / maxBars * np.abs(fft_complex_t))),4)
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #wave_bass_t_crash = shift(wave_bass_t_crash,-5)
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*4
                        calcTrebleVal = int(((maxBars)-(maxBars-(maxBars+(maxBars)))/maxBars))
                        wave_bass_t_snare_thump = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_thump_end_2))*10
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        #print(calcTrebleVal/8)/
                        #wave_bass_t_treble = shift(wave_bass_t_trele,-2)
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        #wave_bass_t_crash[i] /= 2
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_voice[i]+(wave_bass_t_treble[i]*2)+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_mid_2[i]+wave_bass[i])))) * (1-float(args["trebleinitial"]))))
                        treble_i += 1
                    elif(i==(maxBars-5)):
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*10
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*10
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*10
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*10
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*10
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*10
                        #wave_bass_t_crash = shift(wave_bass_t_crash,-5)
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*4
                        calcTrebleVal = int(((maxBars)-(maxBars-(maxBars+(maxBars)))/maxBars))
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*10
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #print(calcTrebleVal/4)
                        #wave_bass_t_treble = shift(wave_bass_t_trele,-2)
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FACTOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        #wave_bass_t_crash[i] /= 2
                        bars[i] = int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_voice[i]+(wave_bass_t_treble[i])+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_mid_2[i])))) * (1-float(args["trebleinitial"]))))
                        treble_i += 1
                    if(i>=(maxBars-2)):
                        #wave_bass_t = smooth1(getTransf/ormedSpectrum(((i+100) / maxBars * np.abs(fft_complex_t))),4)
                        wave_bass = (np.abs(fft_complex)/24)
                        wave_voice = (np.abs(fft_complex_t_voice))
                        #wave_bass_t_low = ((((i))) / maxBars * np.abs(fft_complex_t_low))*10
                        wave_bass_t_mid = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid))*18
                        wave_bass_t_mid_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_mid_2))*18
                        wave_bass_t_snare = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare))*18
                        wave_bass_t_snare_2 = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_snare_2))*18
                        wave_bass_t_snare_thump = ((treble_i)/(maxBars-3)*np.abs(fft_complex_t_snare_thump))*18
                        wave_bass_t_crash = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_crash))*18
                        #fft_complex_t_sizzle
                        wave_bass_t_sizzle = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_sizzle))*18
                        wave_bass_t_treble = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_treble))*18
                        wave_bass_t_b = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_b))
                        #wave_bass_t_high = smooth1(wave_bass_t_low,maxBars)*4
                        #print(wave_bass)
                        #valueMag = wave_bass[mD]
                        #valueMag_t = wave_bass_t[mD]
                        #amplitude[i] = abs((amplitude[i] * SMOOTHING_FA0TOR) + ((abs(((int(valueMag+valueMag_t))))) * (1-SMOOTHING_FACTOR)))
                        #bars[i] = abs((bars[i] * (SMOOTHING_FACTOR-0.10)) + ((abs(((window[i]+int(amplitude[i]))))) * (1-(SMOOTHING_FACTOR-0.10))))
                        #if SMOOTHING_FACTOR != -1:
                            #bars1[i] = int(abs((bars1[i] * (SMOOTHING_FACTOR+0.20)) + ((abs(((wave_bass_t2[i])))) * (1-(SMOOTHING_FACTOR+0.20))))) # wave_bass_t[i]+wave_bass_t2[i]
                        #wave_bass_t_crash[i] /= 2
                        #wave_bass_t_treble[maxBars-1]+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_bass_t_snare_thump[i]
                        #print(wave_bass_t_treble[maxBars-1]+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_bass_t_snare_thump[i])
                        #if((wave_bass_t_treble[maxBars-1]+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_bass_t_snare_thump[i]) > 228000):\
                        wave_bass_t_claps = ((treble_i+3)/(maxBars-3)*np.abs(fft_complex_t_claps_iso))*abs(treble_i)
                        sumAllsHighEnds = 0
                        ampSnare = (bars[i]/bars[maxBars-5])
                        bars[i] = (int(abs((bars[i] * float(args["trebleinitial"])) + (abs(((wave_bass[i]+wave_voice[i]+wave_bass_t_treble[i]+wave_bass_t_mid[i]+wave_bass_t_mid_2[i]+wave_bass_t_snare[i]+wave_bass_t_snare_2[i]+wave_voice[i]+wave_bass_t_snare_thump[i])))) * (1-float(args["trebleinitial"])))))
                        treble_i += 1
            #spectrum = scale(bars, out_range=(-1, maxBars))
            ease += 1
            spectrum = getTransformedSpectrum(bars)
            sum2 = 0
            for i in range(maxBars):
                #if barsSilenced == True:
                #   break
                if(i>=1 and barsSilenced == False):
                    bar1 = [int(1280/12)+int((i+(i*int(args["spacing"])))), int(720/2), int(args["size"]), 4+int(abs((spectrum[i])))]
                    pygame.draw.rect(empty_surface_1,(255,255,255),bar1,0)
                sum1 += np.abs((fft_complex_t_bass[i]))
                if fft_complex_t_kick[i] > 1:
                    isKick = False
                else:
                    isKick = False

                if fft_complex_t_mid_2[i] > 1:
                    isHighhat = True
                else:
                    isHighhat = False
                    
                if fft_complex_t_snare[i] > 1:
                    isSnare = True
                else:
                    isSnare = False

            
                   
                    #surface.blit(image, rect)
            #pygame.draw.rect(screen,(255,255,255),(int(i+(i*10)), int(1000/2), 10, 2+int(abs((spectrum[i])))))
                    #pygame.draw.rect(screen,(0,0,0),(int(i+(i*10)), int(1000/2), 10, 2+int(abs((spectrum[i])))),1)
                    #pygame.draw.rect(screen,(0,0,0),(int(i+(i*10)), int(1000/2), 10, 1+int(abs((spectrum[i])))), 1)
                        #line(surface, color, start_pos, end_pos)
                        #pygame.draw.line(screen, (0,0,0), (int(i+(i*0.5*0.5)), int(512/2)), (int(i+(i*0.5*0.5)), int(512/2)+int(amplitude[i])))
                        #pygame.draw.line(screen, (0,0,0), (int(i+(i*10)), int(512/2)), (int(i+(i*10)+8), int(512/2)-int(amplitude[i])))
                        #pygame.draw.line(background, (0,0,0), (int(i+(i*5)+1), int(512/2)), (int(i+(i*5)+1), int(512/2)-int(amplitude[i])))
                        #pygame.draw.line(background, (0,0,0), (int(i+(i*5)+2), int(512/2)), (int(i+(i*5)+2), int(512/2)-int(amplitude[i])))
                        #pygame.draw.line(background, (0,0,0), (int(i+(i*5)+3), int(512/2)), (int(i+(i*5)+3), int(512/2)-int(amplitude[i])))
            #sum1 = abs(sum1-fft_complex_t_kick[0])
            #pygame.event.wait()
            #clock.tick(30)
            #cv2.imshow("Preview", frame)
            sums = ((sum1/86400))
            checking = int(((sum1/1000)))
            sumItAll = sums
            #print(sumItAll)
            #print(sums)
            #isChecking = True
            #print(sumItAll)
            #sumItAll = abs(sums-fft_complex_t_kick[0])
            #pygame.display.update()
            #out.write(frame)
            #writer.writeFrame(frame)/
            #now = time.time()	
            #pygame.event.wait()
            #out.write(frame)
            #skvideo.io.vwrite(args["output"], frame)
            #cv2.imshow("Preview", frame)
            #frame = cv2.resize(frame,(1280,720))
            #writer.write(frame)
            #clock.tick(60)
            #smoke.render()
            #partsys.render()
            #pygame.transform.flip(empty_surface, False, True)
            #if isChecking == False:
            img_with_flip = pygame.transform.flip(empty_surface_1, False, True)
            screen.blit(img_with_flip, (0, 0))
            pygame.display.flip()
            #clock.tick(60)
            #screen.blit(empty_surface, (0, 0))
            #pygame.transform.flip(screen, True, True)
            #pygame.display.update(barObj)
            #ind = ind + 1
            #time.sleep(0.001)
            #writer.write(data)
            #clock.tick(60)
    stream.stop_stream()
    stream.close()

    p.terminate()        
# do a bit of cleanup
#vs.release() 
#out.release()
print("[INFO] cleaning up...")
#writer.close()
#cv2.destroyAllWindows()
#cv2.destroyAllWindows() 
#vs.stop()
#writer.release()
#f.close()
#result.write_videofile(r".\\out.mp4", fps=60)
pygame.quit()
#pool.close()
