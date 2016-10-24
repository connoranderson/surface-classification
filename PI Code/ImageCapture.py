# Import required Python libraries
import time
import RPi.GPIO as GPIO
from time import sleep
from picamera import PiCamera

 
# ----------- CONSTANT DEFINITIONS -----------------------
# Define GPIO to use on Pi
GPIO_TRIGGER = 23
GPIO_ECHO    = 24
GPIO_FLASH   = 25

CAMERA_CAPTURE_DELAY = 0.025 # Units S
FLASH_ON_DELAY = 0.001
CAMERA_WARM_TIME = 2;


#------------ DISTANCE CAPTURE ROUTINE -----------------------

# Use BCM GPIO references
# instead of physical pin numbers
GPIO.setmode(GPIO.BCM)

print "Ultrasonic Measurement"

# Set pins as output and input
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo

# Set trigger to False (Low)
GPIO.output(GPIO_TRIGGER, False)

# Allow module to settle
time.sleep(0.5)

# Send 10us pulse to trigger
GPIO.output(GPIO_TRIGGER, True)
time.sleep(0.00001)
GPIO.output(GPIO_TRIGGER, False)
start = time.time()

while GPIO.input(GPIO_ECHO)==0:
  start = time.time()

while GPIO.input(GPIO_ECHO)==1:
  stop = time.time()

# Calculate pulse length
elapsed = stop-start

# Distance pulse travelled in that time is time
# multiplied by the speed of sound (cm/s)
distance = elapsed * 34300

# That was the distance there and back so halve the value
distance = distance / 2

print "Distance : %.1f" % distance

# Reset GPIO settings
GPIO.cleanup()

#------------ IMAGE CAPTURE ROUTINE -----------------------

camera = PiCamera()
camera.resolution = (1024, 768)
#camera.start_preview()
# Camera warm-up time
sleep(CAMERA_WARM_TIME)
camera.capture('{timestamp:%H%M%S}-{counter:03d}.jpg') # Individually managed filename
# Give camera time to capture
time.sleep(CAMERA_CAPTURE_DELAY)

#------------ FLASH ON -----------------------
GPIO.output(GPIO_FLASH, True)
# Give flash time to come on
time.sleep(FLASH_ON_DELAY)

#------------ IMAGE CAPTURE ROUTINE -----------------------
camera.capture('{timestamp:%H%M%S}-{counter:03d}.jpg') # Individually managed filename
# Give camera time to capture
time.sleep(CAMERA_CAPTURE_DELAY)

#------------ FLASH OFF -----------------------
GPIO.output(GPIO_FLASH, False)
# Give flash time to come on
time.sleep(FLASH_ON_DELAY)


