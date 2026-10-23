# 🛸 Autonomous Aerial Robotics: Flight Dynamics, Mapping & Exploration

Welcome to the documentation repository for **Autonomous Aerial Robotic Systems** (Unmanned Aerial Vehicles / Drones). This document outlines the fundamental principles governing how aerial robots navigate, perceive their environment, execute Simultaneous Localization and Mapping (SLAM), and perform autonomous exploration in unknown GPS-denied environments.

---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Flight Dynamics & Kinematics (6 DoF)](#flight-dynamics--kinematics-6-dof)"
3. [Perception & Sensor Suite](#perception--sensor-suite)
4. [Simultaneous Localization and Mapping (SLAM)](#simultaneous-localization-and-mapping-slam)
5. [Autonomous Exploration Strategies](#autonomous-exploration-strategies)
6. [System Architecture Pipeline](#system-architecture-pipeline)
7. [Getting Started & Simulation Tools](#getting-started--simulation-tools)

---

## 🚁 Overview

Autonomous aerial systems rely on an integrated pipeline of high-frequency hardware controls and complex software algorithms. Unlike ground robots, aerial robots operate in fully 3D environments with tight energy constraints, fast dynamics, and complex aerodynamics.

```
       +------------------+         +------------------+
       | Perception       | ------> | SLAM &           |
       | (Sensors/LiDAR)  |         | State Estimation |
       +------------------+         +------------------+
                |                            |
                v                            v
       +------------------+         +------------------+
       | Path Planning &  | <------ | Grid Map / Point |
       | Exploration      |         | Cloud Generation |
       +------------------+         +------------------+
                |
                v
       +------------------+
       | Flight Control   | ====> [ Motor Speeds (ESCs) ]
       +------------------+
```

---

## ⚙️ Flight Dynamics & Kinematics (6 DoF)

An aerial robot moves in 3D space with **6 Degrees of Freedom (DoF)**:
* **Translation (Linear):**
  * **$X$ (Surge):** Forward / Backward (*Pitch/Arfagem*)
  * **$Y$ (Sway):** Left / Right (*Roll/Rolagem*)
  * **$Z$ (Heave):** Up / Down (*Throttle/Elevação*)
* **Rotation (Angular):**
  * **Yaw ($\psi$):** Rotation around $Z$-axis (Heading direction)
  * **Pitch ($\theta$):** Tilt along $Y$-axis
  * **Roll ($\phi$):** Tilt along $X$-axis

### Quadrotor Motion Mechanics
Multirotors regulate thrust and body torque by dynamically adjusting the individual angular velocity ($\omega_i$) of their propellers:

| Motion Target | Front Motors | Rear Motors | Left Motors | Right Motors |
| :--- | :--- | :--- | :--- | :--- |
| **Hover / Ascend** | Speed Up ($\uparrow$) | Speed Up ($\uparrow$) | Speed Up ($\uparrow$) | Speed Up ($\uparrow$) |
| **Pitch Forward** | Slow Down ($\downarrow$) | Speed Up ($\uparrow$) | Standard | Standard |
| **Roll Right** | Standard | Standard | Speed Up ($\uparrow$) | Slow Down ($\downarrow$) |
| **Yaw CW / CCW** | Differential torque between CW & CCW rotating pairs |

---

## 👁️ Perception & Sensor Suite

To perceive its surroundings, an autonomous drone utilizes a combination of exteroceptive and proprioceptive sensors:

* **LiDAR (Light Detection and Ranging):** Generates high-density, accurate 3D point clouds independent of ambient light conditions.
* **Stereo Vision & RGB-D Cameras:** Computes depth maps through parallax and triangulates visual features.
* **Inertial Measurement Unit (IMU):** Contains high-rate gyroscopes and accelerometers to measure angular velocities and linear acceleration.
* **Barometers & Ultrasonic Rangefinders:** Provide height-above-ground measurements and altimetry assistance.

---

## 🗺️ Simultaneous Localization and Mapping (SLAM)

In GPS-denied environments (indoors, caves, forests, urban canyons), aerial robots solve the **chicken-and-egg problem**: *computing their unknown location within an unknown map*.

### Key SLAM Implementations
1. **Visual-Inertial Odometry (VIO):** Combines visual features tracked across camera frames with high-frequency IMU data (e.g., ORB-SLAM3, VINS-Mono).
2. **LiDAR SLAM:** Uses Iterative Closest Point (ICP) or NDT algorithms to align sequential point clouds (e.g., FAST-LIO, LOAM).

### Map Representations
* **3D Occupancy Grids (OctoMap / Voxblox):** Divides space into volumetric pixels (*voxels*) categorized as `Free`, `Occupied`, or `Unknown`.
* **Point Clouds:** Raw geometric data points used for fine surface reconstructions and visual inspection.

---

## 🧭 Autonomous Exploration Strategies

Rather than relying on human teleoperation, the robot determines its own flight path to explore unmapped territory.

### Frontier-Based Exploration
1. **Frontier Detection:** Identifies boundaries separating known `Free` space from `Unknown` space.
2. **Information Gain Calculation:** Evaluates frontiers based on expected visibility gain, distance, and energy cost.
3. **Path Planning ($A^*$, $RRT^*$):** Generates collision-free polynomial trajectories to reach selected target frontiers.

---

## 🛠️ System Architecture Pipeline

A typical software stack for an autonomous drone includes:

```bash
├── firmware/           # Low-level control (PX4 / ArduPilot)
├── sensors/            # Drivers for LiDAR, IMU, Depth Cameras
├── localization/       # VIO / LiDAR-Inertial Odometry nodes
├── mapping/            # Voxblox / OctoMap volumetric mapping
├── planning/           # Frontier selection & RRT* path planners
└── controller/         # Trajectory tracking & attitude controller
```

---

## 🚀 Getting Started & Simulation Tools

To experiment with autonomous aerial systems, the following open-source frameworks are recommended:

* **Simulators:** 
  * [Gazebo](http://gazebosim.org/) / [ROS 2](https://docs.ros.org/)
  * [AirSim](https://microsoft.github.io/AirSim/) (Unreal Engine based)
* **Flight Control Software:** [PX4 Autopilot](https://px4.io/) or [ArduPilot](https://ardupilot.org/)
* **Mapping Frameworks:** [OctoMap](https://octomap.github.io/) / [Voxblox](https://github.com/ethz-asl/voxblox)

---

## 📄 License

This repository and documentation are distributed under the **MIT License**.