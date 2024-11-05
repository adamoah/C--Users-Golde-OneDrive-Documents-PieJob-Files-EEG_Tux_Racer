import tensorflow as tf
import pylsl
import numpy as np
from pylsl import StreamInlet, resolve_stream                  
from nltk import flatten
import psutil
import dsp

from pynput.keyboard import Key, Controller
from timeit import default_timer as timer

confidence_threshold = 0.15   
controller = Controller()

def select_key(dir):
    
    if dir == "L":
        controller.press(Key.left)
        controller.release(Key.left)
    elif dir == "R":
        controller.press(Key.right)
        controller.release(Key.right)
    else:
        pass


def features(raw_data):

    implementation_version = 4 # 4 is latest versions

    raw_data = np.array(raw_data)

    axes = ['TP9', 'AF7', 'AF8', 'TP10']                        # Axes names.
    sampling_freq = 250                                         # Sampling frequency of the data.

    #Parameters specific to the spectral analysis DSP block [Default Values].
    scale_axes = 1                                             
    input_decimation_ratio = 1                                  
    filter_type = 'none'                                        
    filter_cutoff = 0                                           
    filter_order = 0                                            
    analysis_type = 'FFT'    
    draw_graphs = False                                  

    # The following parameters only apply to FFT analysis type.  Even if you choose wavelet analysis, these parameters still need dummy values
    fft_length = 64                                             

    # Deprecated parameters. Only for backwards compatibility.  
    spectral_peaks_count = 0                                    
    spectral_peaks_threshold = 0                                
    spectral_power_edges = "0"                                 

    # Current FFT parameters
    do_log = True                                               # Log of the spectral powers from the FFT frames
    do_fft_overlap = True                                       # Overlap FFT frames by 50%.  If false, no overlap
    extra_low_freq = False                                      #Decimate the input window by 10 and perform another FFT on the decimated window.
                                                                # This is useful to extract low frequency data.  The features will be appended to the normal FFT features

    # These parameters only apply to Wavelet analysis type.  Even if you choose FFT analysis, these parameters still need dummy values
    wavelet_level = 2                                           # Level of wavelet decomposition
    wavelet = "rbio3.1"                                         # Wavelet kernel to use

    output = dsp.generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, scale_axes, input_decimation_ratio,
                        filter_type, filter_cutoff, filter_order, analysis_type, fft_length, spectral_peaks_count,
                        spectral_peaks_threshold, spectral_power_edges, do_log, do_fft_overlap,
                        wavelet_level, wavelet, extra_low_freq)


    return output["features"]

# Load the TensorFlow Lite model
model_path = "type the location of your TensorFlow model"            #This is a placeholder 
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print()
print(input_details)
print()
print(output_details)

# Connect to the LSL stream
streams = resolve_stream('type', 'EEG')                         # create a new inlet to read # from the stream
inlet = pylsl.stream_inlet(streams[0])

nr_samples = 1


while "tuxracer.exe" in (i.name() for i in psutil.process_iter()): # check if tuxracer program is running only continue if it is

    back_nr = left_nr = right_nr = 0
    
    for iter in range (nr_samples):
        all_samples = []
        for i in range (2000 // 4):                                                 # 2000 ms = 2 secs, 4 EEG-electrodes (channels)
            sample, timestamp = inlet.pull_sample()
            sample.pop()
            all_samples.append(sample)

        all_samples = flatten(all_samples)                                          
        all_samples = features(all_samples)

        input_samples = np.array(all_samples[:65], dtype=np.float32)
        input_samples = np.expand_dims(input_samples, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_samples)            # input_details[0]['index'] = the index which accepts the input
        interpreter.invoke()                                                        # run the inference

        output_data = interpreter.get_tensor(output_details[0]['index'])            # output_details[0]['index'] = the index which provides the input

        background  = output_data[0][0]
        right       = output_data[0][1]
        left        = output_data[0][2]
        
        if left >= confidence_threshold:
            predicted_key = "L"  # Adjust based on your model's output mapping
            select_key(predicted_key)
        elif right >= confidence_threshold:
            predicted_key = "R"  # Adjust based on your model's output mapping
            select_key(predicted_key)

    #print(f"Left: {left:.8f}  Background: {background:.8f}  Right: {right:.8f} Blink: {blink:.8f}")       # this is used to show the confidence level of each brain activity.
