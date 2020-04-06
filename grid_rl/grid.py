import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)

import time
import gym
import random
from gym import spaces
from gym.utils import seeding, EzPickle
from collections import deque
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


# Routing Optimization Avoiding Obstacle.

FPS = 25
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

# Drone's shape
DRONE_POLY = [
    (-0.3, +0.3), (-0.3, 0), (-0.3, -0.3),
    (+0.3, -0.3), (+0.3, 0), (+0.3, +0.3)
]

OBSTACLE_INIT_VEL = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
                     (-1/np.sqrt(2), -1/np.sqrt(2))]

VIEWPORT_W = 600
VIEWPORT_H = 400

W = int(VIEWPORT_W / SCALE)
H = int(VIEWPORT_H / SCALE)

# Shape of Walls
WALL_POLY = [
    (-50, +20), (50, 20),
    (-50, -20), (50, -20)
]


HORIZON_LONG = [(W, -0.3), (W, 0.3),
                  (-W, -0.3), (-W, 0.3)  ]
VERTICAL_LONG = [ (-0.3, H), (0.3, H),
                  (-0.3, -H), (0.3, -H)]

HORIZON_SHORT = [(W/3, -0.5), (W/3, 0.5),
                  (-W/3, -0.5), (-W/3, 0.5)  ]
                    # up         # right     # down    # left , left_one_third, right_one_third
WALL_INFOS = {"pos": [(int(W/2), int(H)), (int(W), int(H/2)), (int(W / 2), 0), (0, int(H/2))],
              "vertices": [HORIZON_LONG, VERTICAL_LONG, HORIZON_LONG, VERTICAL_LONG]
}



# Initial Position of Drone and Goal which of each chosen randomly among vertical ends.
DRONE_INIT_POS = [(5, 2), (8, 2), (10, 2), (12, 2), (15, 2), (18, 2)]
GOAL_POS = [(2, 11), (3, 11), (5, 11), (7, 11), (9, 11), (10, 11), (11, 11), (13, 11), (14, 11), (15, 11)]



def normalize_position(x):
    y = np.copy(x)
    y[0] = x[0]/W
    y[1] = x[1]/H
    return y


def denormalize_position(x):
    y = np.copy(x)
    y[0] = x[0] * W
    y[1] = x[1] * H
    return y



VERTICAL = 1
HORIZONTAL = 0

OBSTACLE_POSITIONS = [
    [np.array([0.08, 0.2]), np.array([0.6, 0.2]), HORIZONTAL],
    [np.array([0.08, 0.35]), np.array([0.6, 0.35]), HORIZONTAL],
    [np.array([0.08, 0.55]), np.array([0.6, 0.55]), HORIZONTAL],
    [np.array([0.92, 0.25]), np.array([0.85, 0.25]), HORIZONTAL],
    [np.array([0.92, 0.35]), np.array([0.85, 0.35]), HORIZONTAL],
    [np.array([0.2, 0.9]), np.array([0.2, 0.75]), VERTICAL],
    [np.array([0.4, 0.9]), np.array([0.4, 0.75]), VERTICAL],
    [np.array([0.6, 0.9]), np.array([0.6, 0.75]), VERTICAL],
    [np.array([0.8, 0.9]), np.array([0.8, 0.75]), VERTICAL],
]


OBSTACLE_POSITIONS = [[denormalize_position(x[0]), denormalize_position(x[1]), x[2]] for x in OBSTACLE_POSITIONS]


def rotation_4(z):
    x = z[0]
    y = z[1]
    rot = [[x, y], [-x, y], [-x, -y], [x, -y]]
    return rot


class MovingRange(object):
    def __init__(self, start, end, axis):
        assert start <= end
        self.start = start
        self.end = end
        self.axis = axis
        move_direction = np.zeros(2)
        move_direction[self.axis] = 1
        self.move_direction = move_direction

    def out_of_range(self, o):
        if o.position[self.axis] >= self.end:
            return -1
        elif o.position[self.axis] <= self.start:
            return 1
        else:
            return 0

    @classmethod
    def from_metadata(cls, meta):
        axis = meta[2]
        if meta[0][axis] > meta[1][axis]:
            start = meta[1][axis]
            end = meta[0][axis]
        else:
            start = meta[0][axis]
            end = meta[1][axis]
        return cls(start=start, end=end, axis=axis)


def to_rect(obstacle_pos):

    axis = obstacle_pos[2]
    if axis == HORIZONTAL:
        y_range = 0.6
        x_range = np.abs(obstacle_pos[0][axis] - obstacle_pos[1][axis]) + 0.6
        position = [round((obstacle_pos[0][0] + obstacle_pos[1][0])/2), round((obstacle_pos[0][1] + obstacle_pos[1][1])/2)]
        poly = rotation_4([round(x_range/2), y_range/2])
    else:
        y_range = round(np.abs(obstacle_pos[0][axis] - obstacle_pos[1][axis]) + 0.6)
        x_range = 0.6
        position = [round((obstacle_pos[0][0] + obstacle_pos[1][0])/2), round((obstacle_pos[0][1] + obstacle_pos[1][1])/2)]
        poly = rotation_4([x_range/2, round(y_range/2)])

    return position, poly


class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.p2 = point
        self.fraction = fraction
        return 0


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.drone == contact.fixtureA.body or self.env.drone == contact.fixtureB.body:
            # if the drone is collide to something, set game over true
            self.env.game_over = True
            # if the drone collide with the goal, success
            if self.env.goal == contact.fixtureA.body or self.env.goal == contact.fixtureB.body:
                self.env.achieve_goal = True


class NavigationEnvDefault(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    """
    dictionary representation of observation 
    it is useful handling dict space observation, 
    classifying local observation and global observation, 
    lazy evaluation of observation space; whenever we add or delete some observation information   
    """
    observation_meta_data = {
        "position": gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "goal_position": gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "lidar": np.zeros(4),
        "energy": gym.spaces.Box(low=0, high=1, shape=(1, ))
    }

    # meta data keys. It is used to force order to get observation.
    # We may use ordered dict, but saving key with list is more economic
    # rather than create ordered dict whenever steps proceed
    observation_meta_data_keys = ["position", "goal_position", "lidar", "energy"]

    def __init__(self, max_obs_range=3, num_disturb=4, num_obstacle=10, initial_speed=2, tail_latency=5,
                 latency_accuracy = 0.95, obs_delay=3):
        self.seed()
        self.map = np.zeros((int(W+1), int(H+1)))
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, 0))
        self.moon = None
        self.drone = None
        self.obstacle = None
        self.disturbs = []
        self.walls = []
        self.obstacles = []
        self.goal = None
        self.obs_tracker = None
        self.obs_range_plt = None
        self.max_obs_range = max_obs_range
        self.prev_reward = None
        self.num_beams = 16
        self.lidar = None
        self.drawlist = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.tail_latency= tail_latency
        self.num_disturbs = num_disturb
        self.num_obstacles = num_obstacle
        self.dynamics = initial_speed
        self.energy = 1
        self.latency_error = (1 - latency_accuracy)
        self.max_delay = obs_delay
        self.min_speed = 1
        self.max_speed=  5
        self.speed_table = None
        p1 = (0, 0)
        p2 = (W , 0)
        self.sky_polys = [[p1, p2, (p2[0], H), (p1[0], H)]]
        self.reset()

    @property
    def observation_space(self):
        # lidar + current position + goal position + energy + risk

        size = 0
        for k in self.observation_meta_data:
            val = self.observation_meta_data[k]
            if k == 'lidar':
                val = val.flatten()
            size += val.shape[0]
        print(size)
        return spaces.Box(-np.inf, np.inf, shape=(size, ), dtype=np.float32)

    @property
    def action_space(self):
        # Nop, left, right, up, down
        return spaces.Discrete(5)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dict_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        lidar = [l.fraction for l in self.lidar]
        obstacle_speed = [np.linalg.norm(o.linearVelocity)/self.max_speed for o in self.obstacles]
        dict_obs = {
            "position":position,
            "goal_position": goal_position,
            "lidar":lidar,
            "energy":self.energy,
            "obstacle_speed": obstacle_speed
        }
        return dict_obs

    def array_observation(self):
        dict_obs = self.dict_observation()
        obs = []
        for k in self.observation_meta_data_keys:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())
        return np.concatenate(obs)

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.drone)
        self.drone = None
        self._clean_walls(True)
        self.world.DestroyBody(self.goal)
        self.goal = None
        self.world.DestroyBody(self.obs_range_plt)
        self.obs_range_plt = None
        self._clean_obstacles(True)

    def _observe_lidar(self, pos):
        for i in range(self.num_beams):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(i * 2 * np.pi / self.num_beams) * self.max_obs_range,
                pos[1] + math.cos(i * 2 * np.pi / self.num_beams) * self.max_obs_range)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

    def _build_wall(self):
        wall_pos =WALL_INFOS["pos"]
        wall_ver = WALL_INFOS["vertices"]
        for p, v in zip(wall_pos, wall_ver):
            wall = self.world.CreateStaticBody(
                position=p,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=v),
                    density=100.0,
                    friction=0.0,
                    categoryBits=0x001,
                    restitution=1.0,)  # 0.99 bouncy
            )
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.walls.append(wall)

    def _build_obstacles(self):
        for i in range(len(OBSTACLE_POSITIONS)):
            pos = np.random.uniform(low=OBSTACLE_POSITIONS[i][0], high=OBSTACLE_POSITIONS[i][1])
            vel = OBSTACLE_POSITIONS[i][1] - pos

            obstacle = self.world.CreateDynamicBody(
                position=(pos[0], pos[1]),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=0.3, pos=(0, 0)),
                    density=5.0,
                    friction=0,
                    categoryBits=0x001,
                    maskBits=0x0010,
                    restitution=1.0,
                )  # 0.99 bouncy
            )
            obstacle.color1 = (0.7, 0.2, 0.2)
            obstacle.color2 = (0.7, 0.2, 0.2)
            speed = np.random.uniform(low=self.min_speed, high=self.max_speed)
            self.speed_table[i] = speed / self.max_speed
            obstacle.linearVelocity.Set(speed* vel[0], speed* vel[1])
            range_= MovingRange.from_metadata(OBSTACLE_POSITIONS[i])
            setattr(obstacle, "moving_range", range_)
            self.obstacles.append(obstacle)

    def _clean_walls(self, all):
        while self.walls:
            self.world.DestroyBody(self.walls.pop(0))

    def _clean_obstacles(self, all):
        while self.obstacles:
            self.world.DestroyBody(self.obstacles.pop(0))

    def _get_observation(self, position):
        delta_angle = 2* np.pi/self.num_beams
        ranges = [self.world.raytrace(position,
                                      i * delta_angle,
                                      self.max_obs_range) for i in range(self.num_beams)]

        ranges = np.array(ranges)
        return ranges

    def map_init(self, drone, goal):
        drone_x = int(drone[0])
        drone_y = int(drone[1])
        goal_x = int(goal[0])
        goal_y = int(goal[1])
        self.map[drone_x][drone_y] = 1
        self.map[goal_x][goal_y] = 10

    def reset(self):
        self.map = np.zeros((int(W+1), int(H+1)))
        self.game_over = False
        self.prev_shaping = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.speed_table = np.zeros(len(OBSTACLE_POSITIONS))
        # timer
        self.energy = 1
        # clean up objects in the Box 2D world
        self._destroy()
        # create lidar objects
        self.lidar = [LidarCallback() for _ in range(self.num_beams)]
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        # create new world
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        p1 = (1, 1)
        p2 = (W - 1, 1)
        self.moon.CreateEdgeFixture(
            vertices=[p1, p2],
            density=100,
            friction=0,
            restitution=1.0,
        )
        self._build_wall()
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # create obstacles
        self._build_obstacles()

        # create controller object
        drone_pos = DRONE_INIT_POS[np.random.randint(len(DRONE_INIT_POS))]
        self.drone = self.world.CreateDynamicBody(
            position=drone_pos,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x, y) for x, y in DRONE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x003,  # collide all but obs range object
                restitution=0.0)  # 0.99 bouncy
        )

        self.drone.color1 = (0.5, 0.4, 0.9)
        self.drone.color2 = (0.3, 0.3, 0.5)
        # create goal
        goal_pos = GOAL_POS[np.random.randint(len(GOAL_POS))]
        self.goal = self.world.CreateStaticBody(
            position=goal_pos,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x, y) for x, y in DRONE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x002,
                maskBits=0x0010,  # collide only with control device
                restitution=0.0)  # 0.99 bouncy
        )
        self.goal.color1 = (0., 0.5, 0)
        self.goal.color2 = (0., 0.5, 0)

        self.map_init(drone_pos, goal_pos)

        self.obs_range_plt = self.world.CreateKinematicBody(
            position=(self.drone.position[0], self.drone.position[1]),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=np.float64(self.max_obs_range), pos=(0, 0)),
                density=0,
                friction=0,
                categoryBits=0x0100,
                maskBits=0x000,  # collide with nothing
                restitution=0.3)
        )
        self.obs_range_plt.color1 = (0.2, 0.2, 0.4)
        self.obs_range_plt.color2 = (0.6, 0.6, 0.6)
        self.drawlist = [self.drone, self.goal] + self.walls + self.obstacles
        self._observe_lidar(drone_pos)

        return np.copy(self.array_observation())

    def step(self, action: np.ndarray):
        for i in range(len(self.obstacles)):
            o = self.obstacles[i]
            moving_range = o.moving_range.out_of_range(o)
            if moving_range != 0:
                speed = np.random.randint(low=self.min_speed, high=self.max_speed)
                next_velocity = (speed * moving_range) * o.moving_range.move_direction
                o.linearVelocity.Set(next_velocity[0], next_velocity[1])
                self.speed_table[i] = speed / self.max_speed

        pos = np.array(self.drone.position)
        goal_pos = np.array(self.goal.position)
        action = np.array(action, dtype=np.float64)
        self.drone.linearVelocity.Set(action[0], action[1])
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.energy -= 1e-3

        self._observe_lidar(pos)
        reward = 0.0025 - 0.01 * np.linalg.norm(pos - goal_pos)
        done = self.game_over
        if done:
            if self.achieve_goal:
                reward = 100

        if self.energy <= 0:
            done = True
        if done and not self.achieve_goal:
            reward = -10
        info = {
            'success':
                self.achieve_goal,
            'energy':
            self.energy
        }
        return np.copy(self.array_observation()), reward, done, info

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class NavigationEnvGrid(NavigationEnvDefault):
    observation_meta_data = {
        "position": gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "goal_position": gym.spaces.Box(np.array([0, 0]), np.array([W, H]), dtype=np.float32),
        "energy": gym.spaces.Box(low=0, high=1, shape=(1, )),
        "lidar": np.zeros(4)
    }

    observation_meta_data_keys = ["position", "goal_position", "energy", "lidar"]

    def map_wall(self):
        for i in range(int(W+1)):
            for j in range(int(H+1)):
                if i == 0:
                    self.map[i][j] = -1
                if j == 0:
                    self.map[i][j] = -1
                if i == int(W):
                    self.map[i][j] = -1
                if j == int(H):
                    self.map[i][j] = -1

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    def map_obstacles(self, pos, poly):
        x = int(pos[0])
        y = int(pos[1])
        x_range = int(abs(poly[0][0]))
        y_range = int(abs(poly[0][1]))
        for i in range(x-x_range, x+x_range+1):
            for j in range(y-y_range, y+y_range+1):
                self.map[i][j] = -1

    def map_lidar(self, pos):
        x = int(pos[0])
        y = int(pos[1])
        if x == 20 or y == 13 or x == 0 or y == 0:
            return [-1, -1, -1, -1]
        return [self.map[x-1][y], self.map[x+1][y], self.map[x][y-1], self.map[x][y+1]]

    def _build_wall(self):
        wall_pos =WALL_INFOS["pos"]
        wall_ver = WALL_INFOS["vertices"]
        for p, v in zip(wall_pos, wall_ver):
            wall = self.world.CreateStaticBody(
                position=p,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=v),
                    density=100.0,
                    friction=0.0,
                    categoryBits=0x001,
                    restitution=1.0,)  # 0.99 bouncy
            )
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.walls.append(wall)
        self.map_wall()

    def _build_obstacles(self):
        for i in range(len(OBSTACLE_POSITIONS)):
            pos, poly = to_rect(OBSTACLE_POSITIONS[i])
            self.map_obstacles(pos, poly)
            wall = self.world.CreateStaticBody(
                position=pos,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=poly),
                    density=100.0,
                    friction=0.0,
                    categoryBits=0x001,
                    restitution=1.0, )  # 0.99 bouncy
            )
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.obstacles.append(wall)

    def dict_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        dict_obs = {
            "position": position,
            "goal_position": goal_position,
            "energy": self.energy,
            "lidar": self.map_lidar(self.drone.position)
            # "lidar": self.map
        }
        return dict_obs

    def original_observation(self):
        position = normalize_position(self.drone.position)
        goal_position = normalize_position(self.goal.position)
        self._observe_lidar(pos=self.drone.position)
        lidar_data = np.array([l.fraction for l in self.lidar])
        dict_obs = {
            "position": position,
            "goal_position": goal_position,
            "energy": self.energy,
            "lidar": lidar_data
            # "lidar": self.map
        }
        return dict_obs

    def array_observation(self):
        dict_obs = self.dict_observation()
        obs = []
        for k in self.observation_meta_data_keys:
            obs.append(np.asarray(dict_obs[k], dtype=np.float32).flatten())
        return np.concatenate(obs)

    def check_game_over(self):
        pos = self.drone.position
        x = int(pos[0])
        y = int(pos[1])
        if self.map[x][y] == -1 or pos == self.goal.position:
            return True
        else:
            return False

    def check_achieve_goal(self):
        if self.drone.position == self.goal.position:
            return True
        else:
            return False

    def map_update(self, past_pos):
        past_x = int(past_pos[0])
        past_y = int(past_pos[1])
        x = int(self.drone.position[0])
        y = int(self.drone.position[1])
        self.map[past_x][past_y] = 0
        self.map[x][y] = 1

    def step(self, action):
        pos = np.array(self.drone.position)
        # nop, left, right, up, down
        if action == 0:
            self.drone.position.Set(pos[0]-1, pos[1])
        if action == 1:
            self.drone.position.Set(pos[0]+1, pos[1])
        if action == 2:
            self.drone.position.Set(pos[0], pos[1]+1)
        if action == 3:
            self.drone.position.Set(pos[0], pos[1]-1)

        self.world.Step(1.0/FPS, 6 * 30, 2 * 30)
        self.energy -= 1e-3
        pos = np.array(self.drone.position)
        goal_pos = np.array(self.goal.position)

        self.game_over = self.check_game_over()
        self.achieve_goal = self.check_achieve_goal()
        self.map_update(pos)
        reward = 0
        done = self.game_over
        if done:
            if self.achieve_goal:
                reward = 1

        if self.energy <= 0:
            done = True

        info = {
            'success':
                self.achieve_goal,
            'energy':
            self.energy
        }
        return np.copy(self.array_observation()), reward, done, info

    def DFS(self, adj, s, e):
        cost = 0
        tovisit = [s]
        visited = []
        result = []
        while tovisit:
            u = tovisit.pop()
            result.append(u)
            visited.append(u)
            cost += 1
            last_node = True
            for v in adj[u]:
                if v not in result + tovisit:
                    tovisit.append(v)
                    last_node = False
            if u == e:
                print('achieve goal')
                return result, visited, cost
            if last_node == True:
                return_way = visited.pop()
                more_return = True
                while more_return == True:
                    return_way = visited.pop()
                    cost += 1
                    for v in adj[return_way]:
                        if v in tovisit:
                            more_return = False
                visited.append(return_way)
        return result, visited, cost

    def run_DFS(self):
        G = self.init_DFS()
        start = ((H+1) * int(self.drone.position[0])) + int(self.drone.position[1])
        print(int(self.drone.position[0]), int(self.drone.position[1]))
        print(int(self.goal.position[0]), int(self.goal.position[1]))
        end = ((H+1) * int(self.goal.position[0]) + int(self.goal.position[1]))
        result, fast_way, cost = self.DFS(G, start, end)
        return cost

    def init_DFS(self):
        dict = {}
        for i in range(W+1):
            for j in range(H+1):
                if self.map[i][j] != -1:
                    l = []
                    if self.map[i-1][j] != -1:
                        l.append(((H+1) * (i-1) + j))
                    if self.map[i+1][j] != -1:
                        l.append(((H+1) * (i+1) + j))
                    if self.map[i][j-1] != -1:
                        l.append(((H+1) * i + (j-1)))
                    if self.map[i][j+1] != -1:
                        l.append(((H+1) * i + (j+1)))
                    random.shuffle(l)
                    dict[((H+1) * i) + j] = l
        return dict


def to_obs(d):
    obs = []
    for k in observation_meta_data_keys:
        obs.append(np.asarray(d[k], dtype=np.float32).flatten())
    return np.concatenate(obs)


from stable_baselines import PPO2, TRPO, ACKTR
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
if __name__ == "__main__":
    observation_meta_data_keys = ["position", "goal_position", "lidar", "energy"]
    obs_action_set = {"obs": [], "actions":[]}
    env = NavigationEnvGrid()
    model = ACKTR.load("test.zip")
    action_map = {0: np.asarray([-1, 0]),
                  1: np.asarray([1, 0]), 2: np.asarray([0, 1]),
                  3: np.asarray([0, -1])}

    for i in range(5000):
        obs = env.reset()
        done = False
        d_obs = env.original_observation()
        obs_action_set["obs"].append(np.copy(to_obs(d_obs)))
        while not done:
            action, _ = model.predict(obs)
            a_real = action_map[action]
            obs_action_set["actions"].append(np.copy(a_real))

            obs, reward, done, info = env.step(action)
            d_obs = env.original_observation()
            if not done:
                obs_action_set["obs"].append(np.copy(to_obs(d_obs)))
        print(i,"/", 5000)
    obs_action_set["obs"] = np.asarray(obs_action_set["obs"])
    obs_action_set["actions"] = np.asarray(obs_action_set["actions"])
    len_obs = len(obs_action_set["obs"])
    obs_action_set["episode_returns"] = np.zeros(shape=len_obs)
    obs_action_set["rewards"] = np.zeros(shape=len_obs)
    obs_action_set["episode_starts"] = np.zeros(shape=len_obs)
    np.savez("wall_pretraining_set", **obs_action_set)
