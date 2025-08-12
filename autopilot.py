import cv2
import numpy as np
import time
from PIL import Image
import mss
from pynput.keyboard import Key, Listener
import math
import os
from datetime import datetime

try:
    import pydirectinput
    DIRECTINPUT_AVAILABLE = True
    print("Using pydirectinput for game-compatible input")
except ImportError:
    from pynput.keyboard import Controller as pynputController
    DIRECTINPUT_AVAILABLE = False
    print("pydirectinput not available, using pynput (may not work with all games)")
    print("Install with: pip install pydirectinput")

class RacingBot:
    def __init__(self):
        self.running = False
        self.paused = False

        self.minimap_region = {'top': 763, 'left': 77, 'width': 260, 'height': 260}
        self.track_color_lower = np.array([190, 190, 185])
        self.track_color_upper = np.array([220, 220, 220])
        self.player_color_lower = np.array([90, 170, 230])
        self.player_color_upper = np.array([140, 220, 270])

        self.look_ahead_distance = 20
        self.max_turn_angle = 60

        self.pid_Kp = 0.06
        self.pid_Ki = 0.002
        self.pid_Kd = 0.03

        self.angle_throttle_threshold = 40

        self.pid_integral = 0; self.pid_last_error = 0; self.last_time = time.time()

        self.keys_pressed = set(); self.sct = mss.mss()
        if DIRECTINPUT_AVAILABLE:
            pydirectinput.FAILSAFE = False; self.input_method = 'directinput'
        else:
            self.keyboard_controller = pynputController(); self.input_method = 'pynput'

        self.debug_mode = True; self.record_debug = True; self.frame_count = 0
        self.video_writer = None; self.recording_active = False
        self.output_dir = "racing_bot_recordings"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        self.last_w_refresh = time.time()  # Add this line for W key refresh timer
        self.w_refresh_interval = 5  # Refresh W key every 5 seconds

    def process_frame(self):
        """Main processing loop for one frame."""
        frame = self.capture_minimap()
        if frame is None: return

        player_positions = self.detect_player_position(frame)
        track_mask = self.detect_track_boundaries(frame)

        optimal_angle, sample_points = self.calculate_track_direction(frame, player_positions, track_mask)
        final_steering = self.calculate_pid_steering(optimal_angle)

        if not self.paused:
            self.send_controls(final_steering, optimal_angle)

        self.frame_count += 1

        if self.debug_mode:
            debug_frame = self.draw_debug_info(frame, player_positions, final_steering, optimal_angle, sample_points)
            if self.record_debug and self.recording_active:
                self.record_debug_frame(debug_frame, track_mask)

    def send_controls(self, steering, turn_angle):
        """Sends keyboard commands based on steering and turn angle."""
        if self.input_method == 'directinput':
            self._send_controls_directinput(steering, turn_angle)
        else:
            self._send_controls_pynput(steering, turn_angle)

    def _send_controls_directinput(self, steering, turn_angle):
        steering_deadzone = 0.15
        current_time = time.time()

        # Handle W key refresh when going straight
        if abs(steering) <= steering_deadzone and abs(turn_angle) <= self.angle_throttle_threshold:
            if current_time - self.last_w_refresh >= self.w_refresh_interval:
                if 'w' in self.keys_pressed:
                    pydirectinput.keyUp('w')
                    self.keys_pressed.remove('w')
                    time.sleep(0.001)  # 1ms pause
                    pydirectinput.keyDown('w')
                    self.keys_pressed.add('w')
                self.last_w_refresh = current_time

        if abs(turn_angle) > self.angle_throttle_threshold:
            if 'w' in self.keys_pressed:
                pydirectinput.keyUp('w')
                self.keys_pressed.remove('w')
        else:
            if 'w' not in self.keys_pressed:
                pydirectinput.keyDown('w')
                self.keys_pressed.add('w')

        if steering < -steering_deadzone: 
            if 'd' in self.keys_pressed: pydirectinput.keyUp('d'); self.keys_pressed.remove('d')
            if 'a' not in self.keys_pressed: pydirectinput.keyDown('a'); self.keys_pressed.add('a')
        elif steering > steering_deadzone: 
            if 'a' in self.keys_pressed: pydirectinput.keyUp('a'); self.keys_pressed.remove('a')
            if 'd' not in self.keys_pressed: pydirectinput.keyDown('d'); self.keys_pressed.add('d')
        else: 
            if 'a' in self.keys_pressed: pydirectinput.keyUp('a'); self.keys_pressed.remove('a')
            if 'd' in self.keys_pressed: pydirectinput.keyUp('d'); self.keys_pressed.remove('d')

    def _send_controls_pynput(self, steering, turn_angle):
        steering_deadzone = 0.15
        current_time = time.time()

        # Handle W key refresh when going straight
        if abs(steering) <= steering_deadzone and abs(turn_angle) <= self.angle_throttle_threshold:
            if current_time - self.last_w_refresh >= self.w_refresh_interval:
                if 'w' in self.keys_pressed:
                    self.keyboard_controller.release('w')
                    self.keys_pressed.remove('w')
                    time.sleep(0.001)  # 1ms pause
                    self.keyboard_controller.press('w')
                    self.keys_pressed.add('w')
                self.last_w_refresh = current_time

        if abs(turn_angle) > self.angle_throttle_threshold:
            if 'w' in self.keys_pressed:
                self.keyboard_controller.release('w')
                self.keys_pressed.remove('w')
        else:
            if 'w' not in self.keys_pressed:
                self.keyboard_controller.press('w')
                self.keys_pressed.add('w')

        if steering < -steering_deadzone:
            if 'd' in self.keys_pressed: self.keyboard_controller.release('d'); self.keys_pressed.remove('d')
            if 'a' not in self.keys_pressed: self.keyboard_controller.press('a'); self.keys_pressed.add('a')
        elif steering > steering_deadzone:
            if 'a' in self.keys_pressed: self.keyboard_controller.release('a'); self.keys_pressed.remove('a')
            if 'd' not in self.keys_pressed: self.keyboard_controller.press('d'); self.keys_pressed.add('d')
        else:
            if 'a' in self.keys_pressed: self.keyboard_controller.release('a'); self.keys_pressed.remove('a')
            if 'd' in self.keys_pressed: self.keyboard_controller.release('d'); self.keys_pressed.remove('d')

    def calculate_pid_steering(self, optimal_angle):
        current_time = time.time(); dt = current_time - self.last_time
        if dt <= 0: return np.clip(self.pid_last_error * self.pid_Kp, -1, 1)
        error = optimal_angle; self.pid_integral += error * dt
        self.pid_integral = np.clip(self.pid_integral, -1.5, 1.5)
        derivative = (error - self.pid_last_error) / dt
        p_term = self.pid_Kp * error; i_term = self.pid_Ki * self.pid_integral; d_term = self.pid_Kd * derivative
        steering_output = np.clip(p_term + i_term + d_term, -1, 1)
        self.pid_last_error = error; self.last_time = current_time
        return steering_output

    def calculate_track_direction(self, frame, player_positions, track_mask):
        h, w = frame.shape[:2]; player_tip, _ = player_positions; px, py = player_tip
        sample_points, track_scores = [], []
        for angle_offset in range(-90, 91, 10):
            rad = math.radians(angle_offset)
            look_x = int(px + self.look_ahead_distance * math.sin(rad)); look_y = int(py - self.look_ahead_distance * math.cos(rad))
            if 0 <= look_x < w and 0 <= look_y < h:
                track_score = np.mean(track_mask[max(0, look_y-2):min(h, look_y+3), max(0, look_x-2):min(w, look_x+3)]) / 255.0
                sample_points.append((look_x, look_y, track_score, angle_offset))
                if track_score > 0.2: track_scores.append((angle_offset, track_score))
        if not track_scores: return self.pid_last_error * 0.8, sample_points
        weighted_sum = sum(angle * score for angle, score in track_scores); total_weight = sum(score for _, score in track_scores)
        return (weighted_sum / total_weight if total_weight > 0 else 0), sample_points

    def setup_video_recording(self):
        if not self.record_debug: return
        if self.recording_active: self.stop_video_recording()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); filename = f"racing_debug_{timestamp}.mp4"
        self.current_recording_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); frame_width = self.minimap_region['width']; frame_height = self.minimap_region['height']
        self.video_writer = cv2.VideoWriter(self.current_recording_path, fourcc, 30.0, (frame_width * 2, frame_height))
        if self.video_writer.isOpened(): self.recording_active = True; print(f"✅ Recording started: {self.current_recording_path}")
        else: self.recording_active = False; print("❌ Failed to start video recording")

    def stop_video_recording(self):
        if self.video_writer and self.recording_active:
            self.video_writer.release(); self.recording_active = False
            print(f"💾 Recording saved: {os.path.abspath(self.current_recording_path)}")

    def record_debug_frame(self, debug_frame, track_mask):
        if not self.recording_active or self.video_writer is None: return
        try:
            track_mask_color = cv2.cvtColor(track_mask, cv2.COLOR_GRAY2BGR)
            h, w = self.minimap_region['height'], self.minimap_region['width']
            debug_resized = cv2.resize(debug_frame, (w, h)); mask_resized = cv2.resize(track_mask_color, (w, h))
            combined_frame = np.hstack([debug_resized, mask_resized])
            cv2.putText(combined_frame, "DEBUG VIEW", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, "TRACK MASK", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.video_writer.write(combined_frame)
        except Exception as e: print(f"Error recording frame: {e}")

    def draw_debug_info(self, frame, player_positions, final_steering, optimal_angle, sample_points):
        debug_frame = frame.copy(); player_tip, _ = player_positions
        if sample_points:
            for x, y, score, _ in sample_points:
                color = (0, 0, 255) if score < 0.3 else ((0, 255, 255) if score < 0.7 else (0, 255, 0))
                cv2.circle(debug_frame, (x, y), 3, color, -1)
        goal_rad = math.radians(optimal_angle)
        goal_x = int(player_tip[0] + self.look_ahead_distance * math.sin(goal_rad)); goal_y = int(player_tip[1] - self.look_ahead_distance * math.cos(goal_rad))
        cv2.line(debug_frame, player_tip, (goal_x, goal_y), (255, 255, 0), 2)
        bar_x = debug_frame.shape[1] // 2; bar_y = debug_frame.shape[0] - 20
        cv2.rectangle(debug_frame, (bar_x - 50, bar_y - 5), (bar_x + 50, bar_y + 5), (0,0,0), -1)
        cv2.rectangle(debug_frame, (bar_x - 50, bar_y - 5), (bar_x + 50, bar_y + 5), (255,255,255), 1)
        steer_pos = int(bar_x + final_steering * 50); cv2.line(debug_frame, (steer_pos, bar_y - 5), (steer_pos, bar_y + 5), (0, 255, 255), 3)
        steer_text = f"PID Out: {final_steering:.2f}"; angle_text = f"Goal Angle: {optimal_angle:.2f}"
        keys_text = f"Keys: {', '.join(sorted(self.keys_pressed)) or 'None'}"
        cv2.putText(debug_frame, steer_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_frame, angle_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(debug_frame, keys_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return debug_frame

    def capture_minimap(self):
        try:
            screenshot = self.sct.grab(self.minimap_region)
            img = np.array(Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception: return None

    def detect_player_position(self, frame):
        mask = cv2.inRange(frame, self.player_color_lower, self.player_color_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            topmost = tuple(largest_contour[largest_contour[:,:,1].argmin()][0])
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"]); center_y = int(M["m01"] / M["m00"])
                return topmost, (center_x, center_y)
        return (frame.shape[1] // 2, frame.shape[0] // 2), (frame.shape[1] // 2, frame.shape[0] // 2)

    def detect_track_boundaries(self, frame):
        road_mask = cv2.inRange(frame, self.track_color_lower, self.track_color_upper)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    def release_all_keys(self):
        try:
            keys_to_release = list(self.keys_pressed)
            if self.input_method == 'directinput':
                for key in keys_to_release: pydirectinput.keyUp(key)
            else:
                for key in keys_to_release: self.keyboard_controller.release(key)
            self.keys_pressed.clear()
        except Exception as e: print(f"Error releasing keys: {e}")

    def on_key_press(self, key):
        try: k = key.char
        except AttributeError: k = key
        if k == 'q' or k == Key.esc: self.stop()
        elif k == 'p':
            self.paused = not self.paused; print(f"Bot {'paused' if self.paused else 'resumed'}")
            if self.paused: self.release_all_keys()
        elif k == 'r':
            if self.recording_active: self.stop_video_recording()
            else: self.setup_video_recording()

    def start(self):
        print("Starting Racing Bot...")
        print("--- TUNE PID & ANGLE THROTTLE VALUES! ---")
        
        # Add user prompt for debug recording
        if self.debug_mode:
            while True:
                response = input("Enable debug recording? (y/n): ").lower()
                if response in ['y', 'n']:
                    self.record_debug = True if response == 'y' else False
                    break
                print("Please enter 'y' or 'n'")
        
        print("\nControls:\n 'p'-Pause | 'r'-Record | 'q' or 'ESC'-Quit")
        
        # Add countdown before activation
        print("\nPlease switch to your game window...")
        print("Bot will activate in:")
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Bot activated!")
        
        self.running = True
        self.last_time = time.time()
        
        if self.record_debug:
            self.setup_video_recording()
            
        listener = Listener(on_press=self.on_key_press)
        listener.start()
        
        try:
            while self.running:
                start_time = time.time()
                self.process_frame()
                elapsed = time.time() - start_time
                time.sleep(max(0, 1/60 - elapsed))
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.cleanup()

    def stop(self):
        if self.running: print("Stopping Racing Bot..."); self.running = False

    def cleanup(self):
        self.release_all_keys()
        if self.recording_active: self.stop_video_recording()
        print("Racing Bot stopped.")

def main():
    print("=" * 40); print("3D Racing Game Auto-Pilot Bot"); print("=" * 40)
    bot = RacingBot(); bot.start()

if __name__ == "__main__":
    main()
