# 3D Racing Game Auto-Pilot Bot


![AI driving](https://github.com/user-attachments/assets/ed49c35f-2501-49f5-9f59-3d97ed267be4)



This project is a Python-based auto-pilot bot designed to play various 3D racing games. It operates by capturing the in-game minimap, processing the image with OpenCV to identify the track, and using a sophisticated **PID (Proportional-Integral-Derivative) controller** to make smooth, human-like steering decisions. **neural networks are NOT used**

The bot is not a "perfect" driver that follows a pre-defined path. Instead, it reacts dynamically to the visual information from the minimap, making it adaptable to different tracks and some racing conditions.
## Requirements

You can install all necessary Python packages using pip:

```bash
pip install opencv-python numpy pillow pynput mss pydirectinput
```
## How to use it
ok so first of all you should edit the colour information of the road and the player, you can do this by taking a screenshot of the minimap and then using a colour picker website. next you should define the area of where the ai should be looking for the minimap. currently it is set the the bottom left corner of the screen. after you configured everything all you have to do is:
1. **launch the game**
2. **start the race**
3. **alt tab from the game**
4. **launch the script with administrator**
5. **return to the game as fast as you can**

the program is going to record the minimap area and show gizmos. this will help you debug any issues with the Bot's behaviour

## Features

- **Real-time Screen Capture:** Uses `mss` for high-performance screen capturing of the minimap region.
- **Advanced Image Processing:** Leverages `OpenCV` and `NumPy` to create a binary mask of the track, filtering out other UI elements.
- **Intelligent PID Control:** Implements a PID controller for steering, which attempts to prevent the common issue of jerky, oscillatory movements and allows for smooth, proportional turning.
- **Smart Throttle Management:** The bot can release the throttle on sharp turns (based on a tunable angle threshold) to improve handling and prevent understeer, mimicking an advanced driving technique.
- **DirectInput for Game Compatibility:** Uses `pydirectinput` to send keyboard commands, which is more reliable for games than standard automation libraries.
- **Comprehensive Debugging Tools:** records a side-by-side video of the bot's vision (minimap view and track mask) for easy tuning and analysis.

## How It Works

The bot's logic loop is as follows:

1.  **Capture:** Grabs a screenshot of the minimap area you define.
2.  **Process:** Creates a black-and-white mask of the track based on color matching.
3.  **Analyze:** Scans a semi-circle of points in front of the player icon on the minimap to find the average "optimal" direction of the road ahead. This is the **Goal Angle**.
4.  **Decide (Throttle):** If the Goal Angle is steeper than a set threshold, the bot releases the 'W' key to coast into the turn. Otherwise, it accelerates.
5.  **Decide (Steering):** The Goal Angle is fed into the PID controller. The controller calculates a smooth, proportional steering output to correct the car's heading without overshooting.
6.  **Act:** The final throttle and steering decisions are sent to the game as `w`, `a`, and `d` key presses.
7.  **Repeat:** This entire process repeats many times per second.




Have fun!
