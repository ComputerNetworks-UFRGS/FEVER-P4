# FEVER Architecture Tutorial

If you are reading this while not attending a live P4 tutorial class,
see [below](#older-tutorials) for links to information about recently
given live classes.


## Introduction

Welcome to the FEVER Tutorial! We've prepared a set of instructions of how replicate the experiments used in the FEVER Architecture Paper. The first thing you need to do is to change OUTPUT_DIR in the `cpu.ssh` file

1. P4 Tutorials


2. Traffic Generator


3. Metrics Monitor


4. Filtering


5. Heatmap and Graphics


6. ML model


## P4 Tutorials

For the Testbed of these experiments, we use the P4 Tutorials environment. We recommend installing P4 following P4lang instructions. The only difference between this Testbed and P4 tutorials is that we included a file in utils called `cpu.ssh` and modificated the Python script `run_exercise.py` to generate traffic automatically and to collect data.

To run an experiment, simply run:

   ```bash
   exercises/mri> make run
   ```

The experiment will run over one hour.

You will notice several instances of MRI inside `tutorialsN`. These correspond to modified versions of MRI to test the capability of identify modifications in the code. `mrimu` is the `mri.p4` with the addition of some lines of codes with multiplications and conditional branches. `mrimod` adds several registers to the code and `mribalance` is the mri topology running `load_balance.p4`.

## Traffic Generator

Inside `run_exercise.py`, there is a class that was not present in the original P4 Tutorials, which is the `TrafficGenerator`. This class is responsible for creating a mininet flow using `iperf`. There are two versions of this class, one on each folder.

There are two folders: One is `tutorialsA` and the other one is `tutorialsN`. N stands for Normal, we generate a normal traffic of 10MB between the hosts. In `tutorialsA`, we generate a maximum throughput to the h3. You can change these values to fullfill your needs. Just go to the class `TrafficGenerator` class in `run_exercise.py` in the folders of normal tutorial and anomalous tutorial.

The traffic is generate


## Metrics Monitor

To collect the data, we use a bash script called `cpu.ssh` that is inside the utils folder of each tutorial folders. You will need to change the value of `OUTPUT_DIR="/home/matheus/testes/MRI-1hour-1s"`. This is where the switch .csv file is saved in your machine after collecting the data. The .csv file correspond to one switch. Inside `run_exercise.py` we call a function that creates a multithread application where you execute a traffic generator while you collect data using the monitor. There is also inside the file functions that identify the PIDs of the switches. Mininet threats each emulated switch as a single process, we use `ps aux` and unix commands to manipulate the process lists and identify the switches PIDs. This PIDs are paramaters to `cpu.ssh`, and each switch is monitored and generates a .csv file with its metrics.

Keep in mind that there are many metrics collected inside the script that are not passed to the .csv file, but are used to calculate metrics. There is also `TOTAL_CPU_USAGE`, which I believe is not correctly or redundantly calculated, so it wasn't used in the experiments later on. Feel free to add metrics using /proc/ , perf and syscalls to make a more elaborated experiment!

The .csv files name is the PID number of the switch. Generally, the switch number correspond to the PID number's magnitude. So the smaller PID number generated correspond to the switch 1 and the higher PID number correspond to the switch n, the last one.

We suggest to create folders for each experiment. Inside `testes` there are examples of folders that ran different exercises and different types of situations(anomalous traffic, normal traffic, etc).

## Filtering

Inside the `testes` folder there is `filter.py`. You need it to take out some unnecessary commas from the generated .csv file. Before running data analysis or ML models, run this:

	```bash
   	python3 filter.py
	```

Then type the `filename.csv` to filter it.

## Heatmap and Graphics.

There is a python script called `hm_high.py`. It does two things: Generates a heatmap of the first file, then ask a second one. This generates a graphic with two lines, one for a normal traffic flow and one with a modified P4 program running in the normal traffic flow.

	```bash
   	python3 hm_high.py
	```

Feel free to change the graphics and legends for them.


# ML models


The ML models are `lof.py`and `vm.py`. When you ran them, they will ask for three .csv files. One for the anomalous switch file, one for the modified P4 program switch file and one for the normal switch.

This will generate a ML prediction which [0] correspond to normal and [1] to anomalous.

ATTENTION: F1-Score is upside down in the evaluation of the NORMAL switch prediction. It predicts correctly but the F1-Score is calculated incorrectly. To calculate correctly, subtract 1 from the F1-Score. The F1-Score for the anomalous and modified switch are calculated correctly. This will be corrected in future versions.

