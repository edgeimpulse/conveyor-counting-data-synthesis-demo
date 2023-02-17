# Edge Impulse - OpenMV Object Detection Example

import sensor, image, time, os, tf, math, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.5

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')


try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

clock = time.clock()



# Define the top of the image and the number of columns
TOP_Y = 100
NUM_COLS = 5
COL_WIDTH = int(sensor.width() / NUM_COLS)
# Define the factor of the width/height which determines the threshold
# for detection of the object's movement between frames:
DETECT_FACTOR = 1.5

# Initialize variables
count = [0] * NUM_COLS
previous_blobs = [[] for _ in range(NUM_COLS)]

while(True):
    clock.tick()

    img = sensor.snapshot()

    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects

    # Initialize list of current blobs
    current_blobs = [[] for _ in range(NUM_COLS)]

    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class?

        print("********** %s **********" % labels[i])
        for d in detection_list:
            [x, y, w, h] = d.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            print('x %d\ty %d' % (center_x, center_y))
            img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)

            # Check which column the blob is in
            col = int(x / COL_WIDTH)
            # Check if blob is within DETECT_FACTOR*h of a blob detected in the previous frame and treat as the same object
            for blob in previous_blobs[col]:
                if abs(x - blob[0]) < DETECT_FACTOR * (w + blob[2]) and abs(y - blob[1]) < DETECT_FACTOR * (h + blob[3]):
                # Check this blob has "moved" across the Y threshold
                    if blob[1] >= TOP_Y and y < TOP_Y:
                        # Increment count for this column if blob has left the top of the image
                        count[col] += 1
             # Add current blob to list
            current_blobs[col].append((x, y, w, h))

    print(clock.fps(), "fps", end="\n\n")

    # Update previous blobs
    previous_blobs = current_blobs

    # Print count for each column
    print("Count: ", count)
    img.draw_string(5, 2, "Object count: {}".format(sum(count)), color=(255, 255, 0))
    img.draw_string(5, 230, "FPS: {}".format(round(clock.fps())), color=(0, 0, 0))

