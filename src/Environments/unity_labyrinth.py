from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import uuid

def build_unity_labyrinth_env():
    """
    Method to construct the unity environment and set up the necessary
    information channels.
    """
    engine_config_channel = EngineConfigurationChannel()
    custom_side_channel = CustomSideChannel()
    side_channels = {
        'engine_config_channel' : engine_config_channel,
        'custom_side_channel' : custom_side_channel,
    }
    unity_env = UnityEnvironment(side_channels=[engine_config_channel, 
                                                    custom_side_channel])

    env = UnityToGymWrapper(unity_env)
    return env, side_channels

# Create the StringLogChannel class
class CustomSideChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self._observers = []

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # Pass the message on to all subscribed observers
        for obs in self._observers:
            obs.notify(self, msg.read_string())
        # # We simply read a string from the message and print it.
        # print(msg.read_string())

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def subscribe(self, observer):
        self._observers.append(observer)

    def unsubscribe(self, observer):
        self._observers.remove(observer)