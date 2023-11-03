import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.integrate import ode
import numpy as np
import pygame

sin = np.sin
cos = np.cos

# to use this environment, replace the existing cartpole environment from gym
class CartPoleEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        print('initializing')
        self.g = 9.81 # gravity constant
        self.m0 = 1.0 # mass of cart
        self.m1 = 0.5 # mass of pole 1
        self.m2 = 0.5 # mass of pole 2
        self.L1 = 1 # length of pole 1
        self.L2 = 1 # length of pole 2
        self.l1 = self.L1/2 # distance from pivot point to center of mass
        self.l2 = self.L2/2 # distance from pivot point to center of mass
        self.I1 = self.m1*(self.L1^2)/12 # moment of inertia of pole 1 w.r.t its center of mass
        self.I2 = self.m2*(self.L2^2)/12 # moment of inertia of pole 2 w.r.t its center of mass
        
        self.tau = 0.02  # seconds between state updates
        self.counter = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 100000 * 2 * math.pi / 360
        self.x_threshold = 2.4


        #self.render_mode = render_mode

        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.state = None

        self.steps_beyond_terminated = None

        self.viewer = None
        """
        observations:
        1. cart position
        2. pendulum 1 angle
        3. pendulum 2 angle
        4.  cart velocity
        5.  pendulum 1 angular velocity
        6   pendulum 2 angular velocity

        """
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high,)

        # Initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display 

    def step(self, action):
        state = self.state
        x = state.item(0)
        theta = state.item(1)
        phi = state.item(2)
        x_dot = state.item(3)
        theta_dot = state.item(4)
        phi_dot = state.item(5)

        #u = force on cart
        u = action
        self.counter += 1
        
        # (state_dot = func(state))
        def func(t, state, u):
            x = state.item(0)
            theta = state.item(1)
            phi = state.item(2)
            x_dot = state.item(3)
            theta_dot = state.item(4)
            phi_dot = state.item(5)
            state = np.matrix([[x],[theta],[phi],[x_dot],[theta_dot],[phi_dot]]) 
            
            #Taken from https://digitalrepository.unm.edu/cgi/viewcontent.cgi?article=1131&context=math_etds pg 17-20
            d1 = self.m0 + self.m1 + self.m2
            d2 = self.m1*self.l1 + self.m2*self.L1
            d3 = self.m2*self.l2
            d4 = self.m1*pow(self.l1,2) + self.m2*pow(self.L1,2) + self.I1
            d5 = self.m2*self.L1*self.l2
            d6 = self.m2*pow(self.l2,2) + self.I2
            
            D = np.matrix([[d1, d2*cos(theta), d3*cos(phi)], 
                    [d2*cos(theta), d4, d5*cos(theta-phi)],
                    [d3*cos(phi), d5*cos(theta-phi), d6]])
            
            C = np.matrix([[0, -d2*sin(theta)*theta_dot, -d3*sin(phi)*phi_dot],
                    [0, 0, d5*sin(theta-phi)*phi_dot],
                    [0, -d5*sin(theta-phi)*theta_dot, 0]])

            g1 = (self.m1*self.l1 + self.m2*self.L1)*self.g 
            g2 = self.m2*self.l2*self.g    
                    
            G = np.matrix([[0], [-g1*sin(theta)], [-g2*sin(phi)]])
            
            H  = np.matrix([[1],[0],[0]])

            #Identity matrix
            I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            #0 matrix (row, col) 
            #3x3 matrix
            O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            #3x1 matrix
            O_3_1 = np.matrix([[0], [0], [0]])
            
            A = np.bmat([[O_3_3, I],[O_3_3, -np.linalg.inv(D)*C]])
            B = np.bmat([[O_3_1],[np.linalg.inv(D)*H]])
            L = np.bmat([[O_3_1],[-np.linalg.inv(D)*G]])
            state_dot = (A * state) + (B * u) + L  
            return state_dot
        
        solver = ode(func) 
        solver.set_integrator("dop853") # (Runge-Kutta)
        solver.set_f_params(u)

        t0 = 0
        state0 = state
        solver.set_initial_value(state0, t0)
        solver.integrate(self.tau)
        state = solver.y
        
        self.state = state
        
        #limitation for action space
        #Change Theta >10000000 and theta<-100000000 to simulate normal DPIC
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or self.counter > 1000 \
                or theta > 90*2*np.pi/360 \
                or theta < -90*2*np.pi/360 
        done = bool(done)

        cost = 10*normalize_angle(theta) + 10*normalize_angle(phi)
                
        reward = -cost
        
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.matrix([[0],[np.random.uniform(-0.1,0.1)],[0],[0],[0],[0]])
        self.counter = 0
        return self.state

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        screen_width = self.screen_width
        screen_height = self.screen_height

        world_width = self.x_threshold*2
        scale = screen_width/world_width

        carty = 300 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.8
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole2.set_color(.2,.6,.4)
            self.poletrans2 = rendering.Transform(translation=(0, polelen-5))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.poletrans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            self.axle2 = rendering.make_circle(polewidth/2)
            self.axle2.add_attr(self.poletrans2)
            self.axle2.add_attr(self.poletrans)
            self.axle2.add_attr(self.carttrans)
            self.axle2.set_color(.1,.5,.8)
            self.viewer.add_geom(self.axle2)
            
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        state = self.state
        cartx = state.item(0)*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-state.item(1))
        self.poletrans2.set_rotation(-(state.item(2)-state.item(1)))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
     

def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle

def main():
    env = CartPoleEnv()
    def some_random_games_first():
    # Each of these is its own game.
        for episode in range(5):
            env.reset()
            # this is each frame, up to 200...but we wont make it that far.
            for t in range(800):
                # This will display the environment
                # Only display if you really want to see it.
                # Takes much longer to display it.
                env.render()
    
                # This will just create a sample action in any environment.
                # In this environment, the action can be 0 or 1, which is left or right
                action = env.action_space.sample()
    
                # this executes the environment with an action,
                # and returns the observation of the environment,
                # the reward, if the env is over, and other info.
                state, reward, done, info = env.step(action)
                env.step(action)

                if done:
                    break

    some_random_games_first()

def q_learning(number_of_states: int=6, number_of_actions: int=2):
    
    return

if __name__ == "__main__":
    main()