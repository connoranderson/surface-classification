# surface-classification

A Stanford University CS229 machine learning project.

Authors: Connor Anderson, Aaron Manheim, and Will Roderick

PI Code: 
	Directory contains surface-classification.py for running on a Raspberry Pi B board. Captures pictures and integrates with a camera, ultrasonic sensor, and LED to generate data used in machine learning algorithm.

imagepreprocessing:
	Directory for preprocessing images. Contains the following:

	Data:
		Directory containing images and distances.txt (output from raspberry pi module). This is excluded from Git repo for file size considerations.

	Subtraction.m 
		Reads distances.txt, processes each image one at a time, and outputs a feature set representing the whole data set. Outputs to two text files, circleParams.txt (a readable matrix), and matrixOutput (a matrix of constants for easy processing).



