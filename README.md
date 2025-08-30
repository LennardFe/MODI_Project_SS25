# <div align="justify">Project in SS 25 for the module Mobile and Distributed Systems in the Master Digital Sciences</div>

Supervisors: **Prof. Dr. Matthias Böhmer**\
Elaboration by: **Ole Berg, Kristan Böttjer and Lennard Feuerbach**

<div align="justify">
  <p> 
  This repository contains the code and data for the research project focused on developing <strong>gesture-based control systems for IoT objects</strong>, such as smart home lamps. The primary objective is to explore intuitive, movement-based interactions with physical devices using sensor data. The central <strong>research question</strong> guiding this project is: 
  </p>

  <blockquote>
    <strong>
      To what extent can ultra-wideband (UWB) radio and inertial measurement unit (IMU) data be used to reliably distinguish and select between closely spaced physical target objects through deliberate arm movements?
    </strong>
  </blockquote>
</div>

<div align="justify">
  <p> 
For a comprehensive overview of the project, including methodology, results, and conclusions, please refer to this project paper (link will follow). For a quick overview of the work, the abstract is provided here: The growing importance of IoT devices calls for intuitive selection mechanisms that align with natural human gestures. This paper introduces a new method for IoT device selection using deliberate arm movements, achieved by fusing data from IMUs and UWB signals to improve accuracy. Arm gestures are recognized by monitoring acceleration data from the IMU. Device selection is then performed using two methods: first, gyrometer data is used to determine the direction of the user’s arm and calculate the angle between the pointing direction and the position of each anchor; second, UWB signals are used to track how distances between the tag and anchors change during the gesture. If the two methods produce different results, a margin score is used to determine which device was most likely the intended target. To evaluate the proposed method, we implemented a prototype system and conducted a series of controlled experiments. The study design tested target selection under varying spatial and environmental conditions. Results demonstrated that UWB-based selection consistently outperformed IMU-based pointing in terms of accuracy and robustness, though IMU showed situational advantages in cases where anchors were closely spaced. Sensor fusion, while not consistently superior to UWB alone, revealed potential benefits in reducing confusion in challenging configurations.
  </p>
</div>
