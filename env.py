import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


def lidar(position, landmarks, noise_std=1e-3, max_range=np.inf):
    detections = []
    for idx, (mx, my) in enumerate(landmarks):
        dx = mx - position[0]
        dy = my - position[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
 
        if distance <= max_range:
            dist_noisy = distance + np.random.normal(0, noise_std)
            angle_noisy = angle + np.random.normal(0, noise_std)
            detections.append((dist_noisy, angle_noisy, idx + 1))
    return detections


class Point2DEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
        low=np.array([-10.0, -10.0, 0.5, 0.0, -1.0, -1.0], dtype=np.float32),
        high=np.array([10.0, 10.0, 2.0, 0.2, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32)

        self.action_space = spaces.Box(low=5*np.array([-1.0, -1.0], dtype=np.float32),
                                       high=5*np.array([1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        self.goal = np.random.uniform(low=-8.0, high=8.0, size=(2,))
        self.position = None
        self.max_steps = 300
        self.current_step = 0
        self.threshold = 0.1
        self.mass = 1.0
        self.friction_coef = 0.2
        self.dt = 0.2
        self.velocity = np.zeros_like(self.position)
        self.friction = self.friction_coef*self.mass*9.81
        self.landmarks = [tuple(self.goal)]  # Treat goal as the only landmark


        # Visualization setup
        self.fig, self.ax = None, None
        self.agent_dot = None
        self.goal_dot = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.position = np.array([0.0, 0.0])  # start at origin
        self.goal = np.random.uniform(low=-8.0, high=8.0, size=(2,))
        self.landmarks = [tuple(self.goal)]  # Treat goal as the only landmark

        # Randomize dynamics
        self.mass = np.random.uniform(0.5, 1.0)
        self.velocity = np.zeros_like(self.position)
        self.friction_coef = np.random.uniform(0.0, 0.1)
        self.friction = self.friction_coef*self.mass*9.81
        self.current_step = 0

        # Setup matplotlib plot on reset
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(self.observation_space.low[0], self.observation_space.high[0])
            self.ax.set_ylim(self.observation_space.low[1], self.observation_space.high[1])
            self.agent_dot, = self.ax.plot([self.position[0]], [self.position[1]], 'bo', label='Agent')  # blue dot
            self.goal_dot, = self.ax.plot([self.goal[0]], [self.goal[1]], 'r*', markersize=15, label='Goal')
        else:
            self.goal_dot.set_data([self.goal[0]], [self.goal[1]])

            self.ax.legend()
        self._update_plot()

        lidar_obs = lidar(self.position, self.landmarks, noise_std=0.01)
        if lidar_obs:
            distance, angle, _ = lidar_obs[0]
        else:
            distance, angle = 0.0, 0.0

        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        obs = np.array([dx, dy, self.mass, self.friction], dtype=np.float32)

        return obs, {}



    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # === Δυναμική Κίνηση ===
        force = action  # θεώρησέ το ως δύναμη που εφαρμόζεται
        self.position, self.velocity = self.compute_dynamics(force)
        self.position = np.clip(self.position, -10.0, 10.0)
        
        # Μετρήσεις lidar
        lidar_obs = lidar(self.position, self.landmarks, noise_std=0.01)
        if lidar_obs:
            distance, angle, _ = lidar_obs[0]
        else:
            distance, angle = 0.0, 0.0

        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        obs = np.array([dx, dy, self.mass, self.friction], dtype=np.float32)

        # Κατεύθυνση προς τον στόχο
        to_goal = np.array([dx, dy])
        # Reward
        reward = self.compute_reward(action, to_goal, distance)

        self.current_step += 1
        done = distance < self.threshold or self.current_step >= self.max_steps

        return obs, reward, done
    

    def compute_reward(self, action, to_goal, distance):
        to_goal_norm = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / to_goal_norm if to_goal_norm > 1e-6 else np.zeros_like(to_goal)

        # Κατεύθυνση της ενέργειας
        action_norm = np.linalg.norm(action)
        action_dir = action / action_norm if action_norm > 1e-6 else np.zeros_like(action)

        # επιβράβευση κατεύθυνσης αν πάει προς τον στόχο
        direction_reward = np.dot(action_dir, to_goal_dir) * 10

        # Βασική επιβράβευση
        reward = -distance + direction_reward
        if distance < self.threshold:
            reward += 100.0

        return reward

    


    def compute_dynamics(self, force):

        direction = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
        friction = -self.friction * direction
        acceleration = (force+friction) / self.mass
        new_velocity = self.velocity + acceleration * self.dt
        new_position = self.position + self.velocity * self.dt + 0.5 * acceleration * self.dt ** 2

        return new_position, new_velocity

    
    def init_pygame_3d(self):
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)


        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)  #  Enable depth buffering for proper 3D

        gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)  # Farther clip
        glTranslatef(0.0, -3.0, -40.0)  #  Pull the camera back & down
        glRotatef(-60, 1, 0, 0)  #  Tilt the view downward on X-axis
        glRotatef(-35, 0, 0, 1)  #  Rotate around Z-axis for angled view

        # === LIGHTING ===
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        light_pos = [10.0, 10.0, 20.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.85, 0.85, 0.85, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 20.0)

        # === ΦΟΡΤΩΣΗ BACKGROUND ΕΙΚΟΝΑΣ ===
        image_path = "space2.jpg"
        self.background_image = pygame.image.load(image_path)
        self.bg_width, self.bg_height = self.background_image.get_size()
        image_data = pygame.image.tostring(self.background_image, "RGB", 1)

        self.bg_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.bg_width, self.bg_height,
                    0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw_shadow(self, x, y, radius=0.65, slices=20):
        glPushAttrib(GL_ENABLE_BIT)  # αποθηκεύει ρυθμίσεις
        glDisable(GL_LIGHTING)       # απενεργοποίηση φωτισμού για τη σκιά
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glColor4f(0.1, 0.1, 0.1, 0.4)  # Semi-transparent dark grey
        glPushMatrix()
        glTranslatef(x-0.3, y+0.2, 0.01)  # Slightly above ground to avoid z-fighting
        glRotatef(-90, 1, 0, 0)  # Rotate to lay flat on XY plane

        quad = gluNewQuadric()
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()

        glPopAttrib()  # Επαναφορά ρυθμίσεων φωτισμού


    def draw_ground(self, height=1):
        glColor3f(0.1, 0.1, 0.1)  # Light grey

        x1, x2 = -10, 10
        y1, y2 = -10, 10
        z1, z2 = 0, -height  # z1 είναι το πάνω επίπεδο, z2 το κάτω


        # Πάνω επιφάνεια (το "έδαφος")
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(x1, y1, z1)
        glVertex3f(x2, y1, z1)
        glVertex3f(x2, y2, z1)
        glVertex3f(x1, y2, z1)
        glEnd()

        # Μπροστά
        glBegin(GL_QUADS)
        glNormal3f(0, -1, 0)
        glVertex3f(x1, y1, z1)
        glVertex3f(x2, y1, z1)
        glVertex3f(x2, y1, z2)
        glVertex3f(x1, y1, z2)
        glEnd()

        # Δεξιά
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glVertex3f(x2, y1, z1)
        glVertex3f(x2, y2, z1)
        glVertex3f(x2, y2, z2)
        glVertex3f(x2, y1, z2)
        glEnd()


    def draw_background(self):
        # Απενεργοποίηση εφέ φωτισμού και βάθους
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        # Εναλλαγή σε orthographic προβολή (2D)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Σχεδίαση του background image ως 2D texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture_id)

        glColor3f(1.0, 1.0, 1.0)  # Χωρίς χρωματική αλλοίωση

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(0, 1)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

        # Επαναφορά κατάστασης προβολής
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        # Επαναφορά φωτισμού και βάθους για τα υπόλοιπα αντικείμενα
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def draw_cylinder(self, radius=0.4, height=0.4, slices=16):
        quad = gluNewQuadric()

        # Draw base disk
        glPushMatrix()
        gluDisk(quad, 0, radius, slices, 1)

        # Draw cylinder wall
        gluCylinder(quad, radius, radius, height, slices, 1)

        # Draw top disk
        glTranslatef(0, 0, height)
        gluDisk(quad, 0, radius, slices, 1)
        glPopMatrix()


    def render_3d(self):
        if not hasattr(self, 'pygame_initialized'):
            self.init_pygame_3d()
            self.pygame_initialized = True

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.draw_background()

        # Υλικό για έδαφος (ανοιχτό γκρι)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.4, 0.4, 0.4, 1.0])  # Σκούρο γκρι
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.05, 0.05, 0.05, 1.0])           # Ελάχιστη ανακλαστικότητα
        glMaterialf(GL_FRONT, GL_SHININESS, 5.0)                               # Σχεδόν ματ επιφάνεια

        self.draw_ground()

        # Draw shadows
        glDisable(GL_DEPTH_TEST)
        self.draw_shadow(self.position[0], self.position[1])
        self.draw_shadow(self.goal[0], self.goal[1])
        glEnable(GL_DEPTH_TEST)

        # Draw agent (blue cylinder)
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], 0)
        # Μπλε υλικό για τον agent
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.1, 0.1, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

        self.draw_cylinder()
        glPopMatrix()

        # Draw goal (red cylinder)
        glPushMatrix()
        glTranslatef(self.goal[0], self.goal[1], 0)
        # Κόκκινο υλικό για το goal
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1.0, 0.1, 0.1, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.4, 0.4, 0.4, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 30.0)

        self.draw_cylinder()
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)

    
    def _update_plot(self):
        if self.position is not None:
            self.agent_dot.set_data([self.position[0]], [self.position[1]])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        self._update_plot()
        real_dist = np.linalg.norm(self.position - self.goal)
        print(f"Step: {self.current_step} | True Pos: {self.position} | Goal: {self.goal} | Distance: {real_dist:.3f}")
