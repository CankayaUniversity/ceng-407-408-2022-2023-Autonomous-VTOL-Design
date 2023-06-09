# VTOL UAV AUTONOMOUS FLIGHT WITH PX4 AND MAVSDK


This is the guide to install and run the development tools for my Circuit Cellar Magazine article "Quadrotor Autonomous Flight with PX4 and MAVSDK"

--------------------------------------------------------------------------------
## 1. INSTALL UBUNTU 18.04
I used Ubuntu 18.04.4 to debug and test all examples in this article. Any newer 18.04 version should also work. I have tried the examples both natively and also in a virtual machine using VirtualBox in a Windows host.
--------------------------------------------------------------------------------
## 2. INSTALL PYTHON PIP
If you don't have Python 'pip' installed already in your system, run the following command in a terminal window:

	sudo apt install python3-pip
--------------------------------------------------------------------------------
## 4. INSTALL THE PX4 FLIGHT STACK AND JMAVSIM & GAZEBO SIMULATORS

Check the official installation guide at: https://dev.px4.io/v1.9.0/en/setup/dev_env_linux.html

Run the following commands:

	cd ~/  # To change directory to your home directory

	wget https://raw.githubusercontent.com/PX4/Devguide/v1.9.0/build_scripts/ubuntu_sim.sh

	source ubuntu_sim.sh

After the installation finishes, the directory will be automatically changed to: ~/src/Firmware
--------------------------------------------------------------------------------
## 5. COMPILE AND RUN THE GAZEBO SIMULATOR

Check the official guide at: https://dev.px4.io/v1.9.0/en/simulation/gazebo.html

One would think all required packages would be already installed after running the previous step, but it turned not to be the case, at least for me. This is the sequence of compilation errors I got when compiling the simulator for the first time and the solutions. The errors are mostly because of some missing libraries required to compile the simulator. 

To compile the Gazebo simulator, run:

	make px4_sitl gazebo

* Compilation error:
	Python import error:  No module named 'em'

	Required python package empy not installed.

	Please run:
	    pip3 install --user empy
	

	- Solution:
	Run: pip3 install --user empy
	And then run again: make px4_sitl gazebo

* Compilation error:
	Python import error:  No module named 'genmsg'

	Required python package pyros-genmsg not installed.

	Please run:
	    pip3 install --user pyros-genmsg

	- Solution:
	Run: pip3 install --user pyros-genmsg
	And then run again: make px4_sitl gazebo

* Compilation error:
	python import error:  No module named 'toml'

	Required python3 packages not installed.

	On a GNU/Linux or MacOS system please run:
	  sudo pip3 install numpy toml

	On Windows please run:
	  easy_install numpy toml

	- Solution:
	Run: sudo pip3 install numpy toml
	And then run again: make px4_sitl gazebo

* Compilation error:
	ModuleNotFoundError: No module named 'jinja2'

	- Solution:
	Run: pip3 install Jinja2
	And then run again: make px4_sitl gazebo

* Compilation error:
	No package 'gstreamer-1.0' found

	- Solution:
	Run: sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
	And then run again: make px4_sitl gazebo

--------------------------------------------------------------------------------
## 6. COMPILE AND RUN THE JMAVSIM SIMULATOR

Check the official guide at: https://dev.px4.io/v1.9.0/en/simulation/jmavsim.html

This step is optional if you will use just Gazebo simulation. To try also jMAVSim, run the following command:

	make px4_sitl jmavsim

If you are compiling jMAVSim for the first time and before Gazebo, you could get the same errors mentioned above in the Gazebo installation process. Aditionally, youl could get also the following error:

	Inconsistency detected by ld.so: dl-lookup.c: 111: check_match: Assertion `version->filename == NULL || ! _dl_name_match_p (version->filename, map)' failed!

It's a Java version error; jMAVSim uses specifically Java V8. The solution is outlined here: https://github.com/PX4/jMAVSim/issues/96

	- Solution:
	For Ubuntu, you can fall back to Java 8 quite easily:

	sudo apt install openjdk-8-jdk
	sudo update-alternatives --config java # choose option 8
	rm -rf Tools/jMAVSim/out
--------------------------------------------------------------------------------
## 7. INSTALL MAVSDK-Python LIBRARY
Check the official installation guide at: https://github.com/mavlink/MAVSDK-Python

- Python 3.6+ is required (because the wrapper is based on asyncio).
- You may need to run pip3 instead of pip and python3 instead of python, depending of your system defaults.

To install MAVSDK-Python, simply run:

	pip3 install mavsdk

To clone the MAVSDK-Python repository with the basic examples, simply run:

	cd ~/  # To download the repository in your home directory
	git clone https://github.com/mavlink/MAVSDK-Python.git
--------------------------------------------------------------------------------
## 8. UBUNTU QGROUNDCONTROL INSTALLATION

Check the official installation guide at: https://docs.qgroundcontrol.com/en/getting_started/download_and_install.html

Before installing QGroundControl for the first time:

1. On the command prompt enter:
	sudo usermod -a -G dialout $USER
	sudo apt-get remove modemmanager -y
	sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav -y

Logout and login again to enable the change to user permissions.

2. To install QGroundControl for Ubuntu Linux 16.04 LTS or later:
	- Download QGroundControl.AppImage.
	- Install (and run) using the terminal commands:
	  chmod +x ./QGroundControl.AppImage
	  ./QGroundControl.AppImage  (or double click)

--------------------------------------------------------------------------------
## 9. RUN YOUR FIRST MAVSDK-PYTHON EXAMPLES

* On command terminal window 1 run:
	cd ~/src/Firmware

	make px4_sitl gazebo

This will start the simulator in a default global location and with the vehicle's graphical simulation. If you want to run the simulator without the vehicle's graphical simulation, run instead:

	HEADLESS=1 make px4_sitl gazebo

This helps to reduce the use of computation power with slower machines or when using virtualization.

* Open QGroundControl software.

* On command terminal window 2 run:

	cd MAVSDK-Python/examples

	Try your first example:
	python3 takeoff_and_land.py

	Try a second example:
	python3 mission.py

	If you want, try any other example includded in the MAVSDK-Python/examples directory
--------------------------------------------------------------------------------
## 10. RUN A SIMULATION ON A SPECIFIC GLOBAL POSITION

From Google Maps, get the latitude and longitude coordinates of the global position you want the simulation to run on (altitude is optional) and run the following lines in the same command terminal where you will run the simulation. For example, a global position for a soccer field in front of my home is: -17.417267, -66.132685. So I ran the following commands:

	cd ~/src/Firmware

	export PX4_HOME_LAT=-17.417267

	export PX4_HOME_LON=-66.132685

	export PX4_HOME_ALT=2500 # This is optional: altitude above the sea level in meters.

	make px4_sitl_default jmavsim

--------------------------------------------------------------------------------
## 11. RUN IN SIMULATION THE EXAMPLE PRESENTED IN THE ARTICLE

* On command terminal window 1 run:
	cd ~/src/Firmware

	make px4_sitl gazebo

	Or, I you don't want the graphical simulation of the vehicle:
	HEADLESS=1 make px4_sitl gazebo

* Open QGroundControl software.

* On command terminal window 2 run:

	cd <folder-where-you-downloaded-the-example>

	Run the article's custom example:
	python3 write_my_initials.py
