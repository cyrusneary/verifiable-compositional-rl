# %%
from gym_minigrid.minigrid import *
from gym_minigrid.window import Window
import numpy as np

class Maze(MiniGridEnv):
    """
    Maze environment.
    
    This environment is full observation.
    The state is (x, y, dir) where x,y indicate
    the agent's location in the environment and
    dir indicates the direction it is facing.
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(
        self,
        agent_start_states = [(1,1,0)],
        slip_p=0.0,
    ):

    """
    Inputs
    ------
    agent_start_states : list
        List of tuples representing the possible initial states 
        (entry conditions) of the agent in the environment.
    slip_p : float
        Probability with which the agent "slips" on any given action,
        and takes another action instead.
    """

        size = 20
        width = size
        height = size

        self.agent_start_states = agent_start_states
        self.goal_states = [(1, height - 2, 0), (1, height - 2, 1), (1, height - 2, 2), (1, height - 2, 3)]

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
        )

        # Action enumeration for this environment
        self.actions = Maze.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=np.array([0,0,0]),
            high=np.array([self.width, self.height, 3]),
            dtype='uint8'
        )

        self.slip_p = slip_p

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate the rooms
        self.grid.wall_rect(0, 0, 6, 6)
        self.grid.wall_rect(5, 0, 15, 6)
        self.grid.wall_rect(8, 5, 6, 11)
        self.grid.wall_rect(13, 5, 7, 11)
        self.grid.wall_rect(0, 5, 9, 6)
        self.grid.wall_rect(0, 10, 9, 6)

        # Add doors
        self.put_obj(Door('grey', is_open=True), 3, 5)
        self.put_obj(Door('grey', is_open=True), 5, 2)
        self.put_obj(Door('grey', is_open=True), 10, 5)
        self.put_obj(Door('grey', is_open=True), 14, 5)
        self.put_obj(Door('grey', is_open=True), 5, 10)
        self.put_obj(Door('grey', is_open=True), 3, 15)
        self.put_obj(Door('grey', is_open=True), 16, 15)

        # Place a goal square
        for goal_state in self.goal_states:
            self.put_obj(Goal(), goal_state[0], goal_state[1])
        
        # Place dangerous lava
        self.grid.horz_wall(2, 7, 3, obj_type=Lava)
        self.grid.horz_wall(6, 8, 2, obj_type=Lava)
        self.grid.horz_wall(3, 12, 2, obj_type=Lava)
        self.grid.horz_wall(6, 14, 2, obj_type=Lava)

        # Place the agent
        if self.agent_start_states:
            # Uniformly pick from the possible start states
            agent_start_state = self.agent_start_states[np.random.choice(len(self.agent_start_states))]
            self.agent_pos = (agent_start_state[0], agent_start_state[1])
            self.agent_dir = agent_start_state[2]
        else:
            self.place_agent()

        self.mission = "get to the goal square"

    def gen_obs(self):
        """
        Generate the observation of the agent, which in this environment, is its state.
        """
        pos = self.agent_pos
        direction = self.agent_dir
        obs_out = np.array([pos[0], pos[1], direction])
        return obs_out

    def step(self, action):
        """
        Step the environment.
        """
        self.step_count += 1

        reward = 0
        done = False

        info = {
            'task_complete' : False,
            'lava' : False
        }

        # Slip probability causes agent to randomly take the wrong action
        if np.random.rand() <= self.slip_p:
            action = np.random.choice(np.array([0, 1, 2]))

        current_pos = self.agent_pos
        current_cell = self.grid.get(*current_pos)
        if current_cell != None and current_cell.type == 'lava':
            # If the agent is in lava, it can no longer do anything
            action = self.actions.done 

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
                max_distance = np.array([self.width, self.height])
                info['lava'] = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        next_state = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        if next_state in self.goal_states:
            info['task_complete'] = True
            done = True
            reward = 1.0

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, info

    def get_num_states(self):
        return self.width * self.height * 4 # position in the gridworld and also facing direction