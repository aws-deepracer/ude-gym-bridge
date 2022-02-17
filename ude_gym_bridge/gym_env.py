#################################################################################
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          #
#                                                                               #
#   Licensed under the Apache License, Version 2.0 (the "License").             #
#   You may not use this file except in compliance with the License.            #
#   You may obtain a copy of the License at                                     #
#                                                                               #
#       http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                               #
#   Unless required by applicable law or agreed to in writing, software         #
#   distributed under the License is distributed on an "AS IS" BASIS,           #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#   See the License for the specific language governing permissions and         #
#   limitations under the License.                                              #
#################################################################################
"""A class for Gym Environment."""
from typing import Optional, List, Tuple, Union, Any, Iterable

from ude import (
    UDEEnvironment,
    SideChannelObserverInterface, SideChannelData,
    AbstractSideChannel,
    UDEServer,
    UDEStepInvokeType,
    Compression, ServerCredentials
)
from ude_gym_bridge.gym_environment_adapter import GymEnvironmentAdapter

import gym


class GymEnv(SideChannelObserverInterface):
    """
    Gym Environment
    """
    def __init__(self,
                 env_name: str = "CartPole-v0",
                 agent_name: str = "agent0",
                 render: bool = True,
                 step_invoke_type: UDEStepInvokeType = UDEStepInvokeType.WAIT_FOREVER,
                 step_invoke_period: Union[int, float] = 120.0,
                 port: Optional[int] = None,
                 options: Optional[List[Tuple[str, Any]]] = None,
                 compression: Compression = Compression.NoCompression,
                 credentials: Optional[Union[ServerCredentials, Iterable[str], Iterable[bytes]]] = None,
                 auth_key: Optional[str] = None,
                 timeout_wait: Union[int, float] = 60.0,
                 **kwargs):
        """

        Args:
            env_name (str): OpenAI Gym Environment name.
            agent_name (str): Name of agent to use.
            render (bool): the flag to render OpenAI gym environment or not.
            step_invoke_type (const.UDEStepInvokeType):  step invoke type (WAIT_FOREVER vs PERIODIC)
            step_invoke_period (Union[int, float]): step invoke period (used only with PERIODIC step_invoke_type)
            port (Optional[int]): Port to use for UDE Server (default: 3003)
            options (Optional[List[Tuple[str, Any]]]): An optional list of key-value pairs
                                                        (:term:`channel_arguments` in gRPC runtime)
                                                        to configure the channel.
            compression (Compression) = channel compression type (default: NoCompression)
            credentials (Optional[Union[ServerCredentials, Iterable[str], Iterable[bytes]]]): grpc.ServerCredentials,
                the path to certificate private key and body/chain file, or bytes of the certificate private
                key and body/chain to use with an SSL-enabled Channel.
            auth_key (Optional[str]): channel authentication key (only applied when credentials are provided).
            timeout_wait (Union[int, float]): the maximum wait time to respond step request to UDE clients.
            kwargs: Arbitrary keyword arguments for grpc.server
        """
        env = gym.make(env_name)
        self._adapter = GymEnvironmentAdapter(gym_env=env,
                                              agent_name=agent_name,
                                              render=render)
        self._ude_env = UDEEnvironment(ude_env_adapter=self._adapter)
        all_envs = gym.envs.registry.all()
        self._env_ids = [env_spec.id for env_spec in all_envs]
        self._ude_server = UDEServer(ude_env=self._ude_env,
                                     step_invoke_type=step_invoke_type,
                                     step_invoke_period=step_invoke_period,
                                     port=port,
                                     options=options,
                                     compression=compression,
                                     credentials=credentials,
                                     auth_key=auth_key,
                                     timeout_wait=timeout_wait,
                                     **kwargs)
        self._adapter.side_channel.register(self)
        # self._ude_server.side_channel.register(self)

    def start(self) -> None:
        """
        Start UDE Server.
        """
        self._ude_server.start()

    def stop(self) -> None:
        """
        Stop UDE Server.
        """
        self._ude_server.close()

    def spin(self) -> None:
        """
        Spin till UDE Server terminates.
        """
        self._ude_server.spin()

    def on_received(self, side_channel: AbstractSideChannel, key: str, value: SideChannelData) -> None:
        """
        Callback when side channel instance receives new message.

        Args:
            side_channel (AbstractSideChannel): side channel instance
            key (str): The string identifier of message
            value (SideChannelData): The data of the message.
        """
        print("key: ", key, " value: ", value)
        if key == "env":
            if value in self._env_ids:
                new_env = gym.make(value)
                self._adapter.env = new_env


def main():
    gym_env = GymEnv()
    gym_env.start()
    gym_env.spin()


if __name__ == '__main__':
    main()
