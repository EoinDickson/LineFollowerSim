import gym
from gym import spaces
import pygame
import numpy as np
import math

MARGIN = 50
STARTPOS = (300,60)


class LineFollowerEnv(gym.Env):   
    metadata = {"render_modes": ["human", "rgb_array"],"render_fps": 60}

    def __init__(self, render_mode=None):
        super(LineFollowerEnv, self).__init__()

        #  Variables
        self.window = None
        self.clock = None
        self.reward = 0
        self.sensor = [0] * 5
        self.sensor_value = [0] * 5

        # New vars
        self.prev_vl = 0
        self.prev_vr = 0
        self.max_time = 60*50



        # Import Sprites
        self.player = pygame.sprite.GroupSingle(Robot(STARTPOS,"imgs/robot.png",0.01*3779.52))
        self.obstacle = pygame.sprite.GroupSingle(Track("imgs/track_advanced.svg",50) )
        for i in range(5): self.sensor[i] = pygame.sprite.GroupSingle(Sensor(self.player.sprite,"imgs/RedSensor.png",1+i))

        #  Window Dimensions
        self.width = self.obstacle.sprite.image.get_width() + MARGIN
        self.height = self.obstacle.sprite.image.get_height() + MARGIN


        # Observation Space
        # self.observation_space = spaces.Box(low =-10000,high = +10000,shape=(5+5,),dtype=np.float32)
        self.observation_space = spaces.Box(low =0,high = 1,shape=(5,),dtype=np.float32)


        # Action Space
        # self.action_space = spaces.Discrete(6)
        # New Action Space
        self.action_space = spaces.Discrete(10)
        # self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Checks that render_mode is "None" or equal to one of the rendoer modes in metadata
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self,action):
        self.max_time -= 1
        # Take action from action space
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        # Update Sprites
        self.player.sprite.update(action)
        for i in range(5): self.sensor_value[i] = self.sensor[i].sprite.update(self.player.sprite,self.obstacle)

        # Check if Done
        if any(self.sensor_value) :
            terminated = False
        else:
            terminated = True
            
        # Reward Calculation
        abs_v = abs(self.player.sprite.vr + self.player.sprite.vl )


        if self.reward < -10:
            terminated = True

        if terminated == False and (abs_v > 0):
            self.reward += 1
        elif terminated == False and (self.player.sprite.vr + self.player.sprite.vl <= 0):
            self.reward -= 5

        if terminated == True:
            self.reward -= 1000

        if self.prev_vl == self.player.sprite.vl and self.prev_vr == self.player.sprite.vr:
            self.reward += .1
        else:
            self.prev_vl = self.player.sprite.vl
            self.prev_vr = self.player.sprite.vr

        if self.sensor_value[2]:
            self.reward += .2
        


        if self.max_time == 0:
            terminated = True

        

        
        # Get Observation
        self.observation = self._get_obs()

        # Render
        if self.render_mode == "human":
            self._render_frame()

        return self.observation, self.reward, terminated, {}

    def reset(self, seed=None,  options=None):
        super().reset(seed=seed)

        # Reset location of Car Sprite
        self.player.sprite.x = STARTPOS[0]
        self.player.sprite.y = STARTPOS[1]
        self.player.sprite.theta = 0
        self.player.sprite.vl = 0.01*self.player.sprite.m2p
        self.player.sprite.vr = 0.01*self.player.sprite.m2p

        # Reset Time
        self.max_time = 60*50

        # Update Sensors
        for i in range(5): self.sensor_value[i] = self.sensor[i].sprite.update(self.player.sprite,self.obstacle)


        # Reset reward
        self.reward = 0

        # Get Observation
        self.observation = self._get_obs()

        self.prev_vl = 0
        self.prev_vr = 0

        # If render mode is "human" render the frames
        if self.render_mode == "human":
            self._render_frame()

        return self.observation
    
    def _get_obs(self):
        # obs = [self.player.sprite.x, 
        #         self.player.sprite.y, 
        #         self.player.sprite.theta, 
        #         self.player.sprite.vl,
        #         self.player.sprite.vr,
        obs = [ self.sensor_value[0],
                self.sensor_value[1],
                self.sensor_value[2],
                self.sensor_value[3],
                self.sensor_value[4]]

        return np.array(obs,dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.width, self.height))

        canvas.fill((255, 255, 255))

        # Draw Sprites
        self.obstacle.draw(canvas)

        self.player.draw(canvas)
        for i in range(5): self.sensor[i].draw(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class Robot(pygame.sprite.Sprite):
    def __init__(self,startpos,robotImg,width):
        super().__init__()
        self.m2p = 3779.52 # meters to pixels
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.vl = 0.01*self.m2p
        self.vr = 0.01*self.m2p
        #Graphics
        self.image = pygame.image.load(robotImg)
        self.orginalImage = self.image
        self.rect = self.image.get_rect(center=(self.x,self.y))
        self.mask = pygame.mask.from_surface(self.image)


    def update(self,event):
        print("Action Value : ")
        print(event)

        # match event:
        #     case 0:
        #         self.vl = -0.02*self.m2p
        #     case 1:
        #         self.vl = -0.01*self.m2p            
        #     case 2:
        #         self.vl = 0
        #     case 3:
        #         self.vl = 0.01*self.m2p
        #     case 4:
        #         self.vl = 0.02*self.m2p
        #     case 5:
        #         self.vr = -0.02*self.m2p
        #     case 6:
        #         self.vr = -0.01*self.m2p
        #     case 7:
        #         self.vr = 0        
        #     case 8:
        #         self.vr = 0.01*self.m2p
        #     case 9:
        #         self.vr = 0.02*self.m2p
        speed = 0.02*self.m2p
        match event:
            case 0:
                self.vl = 0
            case 1:
                self.vl = .25*speed
            case 2:
                self.vl = .5*speed
            case 3:
                self.vl = .75*speed
            case 4:
                self.vl = speed
            case 5:
                self.vr = 0
            case 6:
                self.vr = .25*speed
            case 7:
                self.vr = .5*speed
            case 8:
                self.vr = .75*speed
            case 9:
                self.vr = speed

        # self.vl = event[0] * 30
        # self.vr = event[1] * 30
        self.x +=((self.vl+self.vr)/2)*math.cos(self.theta)*(1/10)#Time
        self.y -=((self.vl+self.vr)/2)*math.sin(self.theta)*(1/10)#Time
        self.theta += (self.vr-self.vl)/self.w*(1/10)#Time

        self.image = pygame.transform.rotate(self.orginalImage, math.degrees(self.theta))
        self.rect = self.image.get_rect(center=(self.x,self.y))
class Track(pygame.sprite.Sprite):
    def __init__(self,trackImg,margin):
        super().__init__()
        self.image = pygame.image.load(trackImg)
        self.mask = pygame.mask.from_surface(self.image)
        self.olist = self.mask.outline()
        self.rect = self.image.get_rect(center = (self.image.get_width()/2,self.image.get_height()/2))
class Sensor(pygame.sprite.Sprite):
    def __init__(self,car,image,id):
        super().__init__()
        self.image = pygame.image.load(image)
        self.image.fill('red')
        self.orginalImage = self.image
        self.id = id
        
        self.mask = pygame.mask.from_surface(self.image)
        self.x = STARTPOS[0]+ 40
        self.y = STARTPOS[1]+ 40
        self.rect = self.image.get_rect(center=(self.x,self.y))

    def update(self,car,obsticle):
        X = car.x + math.cos(car.theta)*40 
        Y = car.y - math.sin(car.theta)*40 

        self.x = X + math.cos(car.theta + math.pi/2)*33 - math.cos(car.theta + math.pi/2)*11*self.id
        self.y = Y - math.sin(car.theta + math.pi/2)*33 + math.sin(car.theta + math.pi/2)*11*self.id


        self.image = pygame.transform.rotate(self.orginalImage, math.degrees(car.theta))
        self.rect = self.image.get_rect(center=(self.x,self.y))
        if pygame.sprite.spritecollide(self,obsticle,False, pygame.sprite.collide_mask):
            self.orginalImage = pygame.image.load("imgs/GreenSensor.png")
            return True
        else:
            self.orginalImage = pygame.image.load("imgs/RedSensor.png")
            return False

    
    def line(self,image):
        self.orginalImage = pygame.image.load(image)
    
    def on_line(self,obsticle):
        if pygame.sprite.spritecollide(self,obsticle,False, pygame.sprite.collide_mask):
            return 1
        else:
            return 0
