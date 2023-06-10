<h1 align="center">
   TALIA Autonomous UAV VTOL Design 
</h1>
<p align="center">
    <img src="https://github.com/CankayaUniversity/ceng-407-408-2022-2023-Autonomous-VTOL-Design/assets/62307840/7c465df4-5998-4ee1-8bb8-e1ff420f4b78">
</p>
<p align="center">
  Welcome to Autonomous UAV VTOL Design information page. 
</p>
We are pleased to introduce you our Autonomous UAV VTOL design project Talia. This project is a multidisciplinary project prepared by the joint work of mechanical engineering, mechatronics engineering and computer engineering departments. 

The aim of the project is to transform the Unmanned Aerial Vehicles (UAV), which are widely used around the world today, into a design that combines drone motion technologies (VTOL) with autonomous flight features and to improve their features.
As it is known, Unmanned Aerial Vehicles carry out their duties in many different areas. We worked on more than one area in our project, where we aimed to improve the mobility and software of Unmanned Aerial Vehicles, which have many different usage areas such as Health, Military, Education, Transportation and Travel.
<h2 align="center">
   Topics
</h2>

Topics covered in this project are listed below:
* Autonomous flight.
* Vertical Takeoff and landing.
* Flight mission planning.
* Object detection.
* Object Tracking.
* Real-time streaming.

<h2 align="center">
  About Talia
</h2>
  
Talia is a Autonomous Unmanned Aerial Vehicle (UAV) design with Vertical Takeoff and Landing (VTOL) characteristics. At the hardware part, it has 4 side motors for Vertical Takeoff and Landing, and 1 rear motor for forward movement. At the frontside of the aircraft, there is one camera module, one telemeter module, one GPS module and one pitot tube. At the software of the project, we worked at several parts like object detection, pbject tracking, real-time streaming and autonomous flight. 

* Object Detection: In the object detection section, we perform real-time object detection by processing the images taken from the front camera of the aircraft during the flight with the artificial intelligence model we trained. Especially for object detection, one of the tasks we set was vehicle detection. For this reason, we created a custom dataset and trained our model according to our needs using YOLOv4 Tiny.

* Object Tracking: In this section, we are talking about the process of tracking our detected object. The tracking algorithm we created, with the help of a square drawn in the middle of the camera screen and a point aligned to the middle of this frame, contains information about the position of the tracked object on the screen, whether our plane is following the object, and the distance of the object from the center.

* Mission Planning and Autonomous Flight: We used the ground station management application QGroundController for the task determination part. The reason we prefer this application is that it is easy and practical to interact with the Pixhawk flight control board in the aircraft. We used the Gazebo simulation application to improve, visualize and add variety to autonomous drone commands and route use.

* Real-time Streaming: We used the real-time broadcasting feature to obtain in-flight images and flight data of the aircraft. We used Real Time Streaming Protocol for this feature. As a result, we can instantly get the information of the detected and tracked image from the ground station.

<h2 align="center">
  About Works
</h2>

* Preparing Custom Dataset : There were some prerequisites necessary for us to fulfill the features we set in the software part. One of them was the deep learning model we had to use for image processing operations. For this reason, we used YOLOv4 Tiny to train our deep learning model. Since our dataset was insufficient, we had to use a custom dataset.
<p align="center">
   <img src="https://github.com/CankayaUniversity/ceng-407-408-2022-2023-Autonomous-VTOL-Design/assets/62307840/50b1bb1a-3058-4f21-9f99-0d09a770fb28" width="200" height="200">
   <img src="https://github.com/CankayaUniversity/ceng-407-408-2022-2023-Autonomous-VTOL-Design/assets/62307840/ff8ff04e-8179-44bd-8f77-236b5c91178c" width="200" height="200">
<p/>

* Object Detection : 



