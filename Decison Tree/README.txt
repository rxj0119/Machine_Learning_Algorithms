Name: Reyansh Jain
ID: 1001670119

Programming Language Used: Python 3.6 :: Anaconda, Inc.
--------------------------

----------------
CODE STRUCTURE
-----------------
FOR DecisionTress.py 

This program gives the test accuracy and train accuracy of the data set provided. 

The Odor attributed has been omitted in running of the program since, it is has the 0 entropy ( E[Odor] = 0) and maximum gain ( G[Odor] = 1). The odor attributes makes the other attributes irrelevant. 
	-----------------
	DataSet Structure
	-----------------
	1. CSV format.
	2. Training data name -> MushroomTrain.csv
	3. Testing data name -> MushroomTest.csv
	4. No attribute headers should be given. 

The attributes considered are -> 'Shape', 'Surface', 'Color', 'Bruises' (in this very order)

-----------
How to Run:
-----------

Python3 <file_name>.py <test_data_path>.csv <train_data_path>.csv

Example: python3 /Users/shreyashshrivastava/Desktop/Decison\ Tree/DecisonTress.py /Users/shreyashshrivastava/Desktop/Decison\ Tree/MushroomTest.csv /Users/shreyashshrivastava/Desktop/Decison\ Tree/MushroomTrain.csv 
Test accuracy =  0.56
Train accuracy =  0.6565656565656566
	


