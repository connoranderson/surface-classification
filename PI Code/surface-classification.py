# Import required Python libraries
import time
import RPi.GPIO as GPIO
from time import sleep
from picamera import PiCamera
import datetime as dt
import subprocess
 
# ----------- CONSTANT DEFINITIONS -----------------------
# Define GPIO to use on Pi

GPIO_SHUTDOWN = 18
GPIO_CLASSIFICATION = 22
GPIO_TRIGGER = 23
GPIO_ECHO    = 24
GPIO_FLASH   = 25
GPIO_CAPTURE = 26

CAMERA_CAPTURE_DELAY = 0.025 # Units S
FLASH_ON_DELAY = 0.001
CAMERA_WARM_TIME = 2
ULTRASONIC_TURN_ON_DELAY = 0.5

# ----------- Initialization -----------------------

# Initialize camera
camera = PiCamera()
camera.resolution = (1024, 768)


# Initialize Ultrasonic Sensor

# Use BCM GPIO references
# instead of physical pin numbers
GPIO.setmode(GPIO.BCM)

print("Ultrasonic Measurement")

# Set pins as output and input
GPIO.setup(GPIO_SHUTDOWN,GPIO.IN)
GPIO.setup(GPIO_CLASSIFICATION,GPIO.IN)
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo
GPIO.setup(GPIO_CAPTURE,GPIO.IN, GPIO.PUD_UP)   # Button
GPIO.setup(GPIO_FLASH,GPIO.OUT)   # Flash



# TODO wait until button is pressed
while True:

	if GPIO.input(GPIO_SHUTDOWN)==0:
		command = 'sudo shutdown -h now'
		subprocess.call([command], shell=True)



	if GPIO.input(GPIO_CAPTURE)==0: # Ready to take picture
                print("Button Pressed")

		# loop the following and then wait again


		#------------ DISTANCE CAPTURE ROUTINE -----------------------
        time.sleep(.5);
          
        # Set trigger to False (Low)
        GPIO.output(GPIO_TRIGGER, False)

        # Allow module to settle
        time.sleep(ULTRASONIC_TURN_ON_DELAY)

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

        print("Distance : %.1f" % distance)


		#------------ IMAGE CAPTURE ROUTINE -----------------------
                
        camera.start_preview()
        # Camera warm-up time
        sleep(CAMERA_WARM_TIME)
        curTime = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S ')
        filename1 = curTime + 'noflash.jpg'
        camera.capture(filename1) # Individually managed filename
        # Give camera time to capture
        time.sleep(CAMERA_CAPTURE_DELAY)

		#------------ FLASH ON -----------------------
		GPIO.output(GPIO_FLASH, True)
		# Give flash time to come on
		time.sleep(FLASH_ON_DELAY)

		#------------ IMAGE CAPTURE ROUTINE -----------------------
		camera.start_preview()
        # Camera warm-up time
        sleep(CAMERA_WARM_TIME)
        filename2 = curTime + 'wflash.jpg'
        camera.capture(filename2) # Individually managed filename
        # Give camera time to capture
        time.sleep(CAMERA_CAPTURE_DELAY)
        camera.stop_preview()

		#------------ FLASH OFF -----------------------
		GPIO.output(GPIO_FLASH, False)
		# Give flash time to turn off
		time.sleep(FLASH_ON_DELAY)

		classification = GPIO.input(GPIO_CLASSIFICATION)


                # Save Data to TXT file
                with open("distanceData.txt","a") as myfile:
                        myfile.write(curTime + ':  Distance: ' + str(round(distance,1)) + '  Classification:' + str(classification) + "\n")


                # Move Images to Data Folder
                command1 ="mv '" + filename1 + "' home/pi/Desktop/CS229\ Project/Data/"
                command2 ="mv '" + filename2 + "' home/pi/Desktop/CS229\ Project/Data/"
                print(command1)
                subprocess.call([command1], shell=True)
                subprocess.call([command2], shell=True)
                        
		
	else:

		time.sleep(0.5)


