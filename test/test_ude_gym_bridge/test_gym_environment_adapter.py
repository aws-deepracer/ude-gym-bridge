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
from unittest import mock, TestCase
from unittest.mock import patch, MagicMock

from ude_gym_bridge.gym_environment_adapter import GymEnvironmentAdapter

from gym.spaces.space import Space


@mock.patch("gym.make")
class GymEnvironmentAdapterTest(TestCase):
    def setUp(self) -> None:
        pass

    def test_initialization(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value
        with patch("ude_gym_bridge.gym_environment_adapter.SingleSideChannel") as side_channel_mock:
            gym_env_adapter = GymEnvironmentAdapter(env_name, agent_name="agent")
            side_channel_mock.assert_called_once()
            gym_env_mock_obj.reset.assert_called_once()
            assert gym_env_adapter._agent_name == "agent"
            assert gym_env_adapter.env == gym_env_mock_obj

    def test_initialization_without_agent_name(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value
        gym_env_adapter = GymEnvironmentAdapter(env_name)

        gym_env_mock_obj.reset.assert_called_once()
        assert gym_env_adapter._agent_name == "agent0"
        assert gym_env_adapter.env == gym_env_mock_obj

    def test_setters(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value
        gym_env_adapter = GymEnvironmentAdapter(env_name)

        gym_env_mock_obj.reset.assert_called_once()
        assert gym_env_adapter._agent_name == "agent0"
        assert gym_env_adapter.env == gym_env_mock_obj

        new_gym_name = "new_test_env"
        new_env_mock = MagicMock()
        gym_make_mock.return_value = new_env_mock

        gym_env_adapter.env_name = new_gym_name
        assert gym_env_adapter.env_name != new_gym_name
        assert gym_env_adapter.env != new_env_mock
        gym_env_adapter.reset()

        gym_env_mock_obj.close.assert_called_once()
        new_env_mock.reset.assert_called_once()
        assert gym_env_adapter.env_name == new_gym_name
        assert gym_env_adapter.env == new_env_mock

    def test_step(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value

        agent_name = "agent0"

        agent_action = 1

        action_dict = {agent_name: agent_action}

        next_state = "next_state"
        done = False
        reward = 42
        info = {}
        gym_env_step_return = (next_state, reward, done, info)
        gym_env_mock_obj.step.return_value = gym_env_step_return

        gym_env_adapter = GymEnvironmentAdapter(env_name,
                                                agent_name=agent_name)
        ret_step_val = gym_env_adapter.step(action_dict=action_dict)

        expected_return = (
            {agent_name: next_state},
            {agent_name: reward},
            {agent_name: done},
            {agent_name: agent_action},
            info
        )
        assert ret_step_val == expected_return
        gym_env_mock_obj.step.assert_called_once_with(agent_action)

    def test_reset(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value

        agent_name = "agent0"

        next_state = "next_state"

        gym_env_mock_obj.reset.return_value = next_state
        gym_env_adapter = GymEnvironmentAdapter(env_name,
                                                agent_name=agent_name)
        ret_reset_val = gym_env_adapter.reset()

        expected_return = (
            {agent_name: next_state},
            {}
        )
        assert ret_reset_val == expected_return
        assert gym_env_mock_obj.reset.call_count == 2

    def test_close(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value
        gym_env_adapter = GymEnvironmentAdapter(env_name)
        gym_env_adapter.close()
        gym_env_mock_obj.close.assert_called_once()

    def test_observation_space(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value
        agent_name = "agent0"

        observation_space = Space([3, 4])

        gym_env_mock_obj.observation_space = observation_space
        gym_env_adapter = GymEnvironmentAdapter(env_name,
                                                agent_name=agent_name)
        ret_observation_space_val = gym_env_adapter.observation_space

        expected_return = (
            {agent_name: observation_space}
        )
        assert ret_observation_space_val == expected_return

    def test_action_space(self, gym_make_mock):
        env_name = "test_env"
        gym_env_mock_obj = gym_make_mock.return_value
        agent_name = "agent0"

        action_space = Space([3, 4])

        gym_env_mock_obj.action_space = action_space
        gym_env_adapter = GymEnvironmentAdapter(env_name,
                                                agent_name=agent_name)
        ret_action_space_val = gym_env_adapter.action_space

        expected_return = (
            {agent_name: action_space}
        )
        assert ret_action_space_val == expected_return

    def test_side_channel(self, gym_make_mock):
        env_name = "test_env"
        with patch("ude_gym_bridge.gym_environment_adapter.SingleSideChannel") as side_channel_mock:
            gym_env_adapter = GymEnvironmentAdapter(env_name)
            side_channel_mock.assert_called_once()
            assert gym_env_adapter.side_channel == side_channel_mock.return_value

    def test_on_received(self, gym_make_mock):
        env_name = "CartPole-v0"
        gym_env_mock_obj = gym_make_mock.return_value
        gym_env_adapter = GymEnvironmentAdapter(env_name)

        assert gym_env_adapter.env == gym_env_mock_obj

        new_env_mock = MagicMock()
        gym_make_mock.return_value = new_env_mock

        assert gym_env_adapter.env != new_env_mock

        new_env_name = "CartPole-v1"

        gym_env_adapter.on_received(side_channel=gym_env_adapter.side_channel,
                                    key="env",
                                    value=new_env_name)
        assert gym_env_adapter.env == gym_env_mock_obj
        assert gym_env_adapter.env != new_env_mock
        assert gym_env_adapter.env_name == env_name

        gym_env_adapter.reset()
        gym_env_mock_obj.close.assert_called_once()
        assert gym_env_adapter.env == new_env_mock
        assert gym_env_adapter.env_name == new_env_name
        new_env_mock.reset_assert_called_once()

    def test_on_received_bad_env_name(self, gym_make_mock):
        env_name = "CartPole-v0"
        gym_env_mock_obj = gym_make_mock.return_value
        gym_env_adapter = GymEnvironmentAdapter(env_name)

        assert gym_env_adapter.env == gym_env_mock_obj

        new_env_mock = MagicMock()
        gym_make_mock.return_value = new_env_mock

        assert gym_env_adapter.env != new_env_mock

        new_env_name = "bad-test-env"

        gym_env_adapter.on_received(side_channel=gym_env_adapter.side_channel,
                                    key="env",
                                    value=new_env_name)
        assert gym_env_adapter.env == gym_env_mock_obj
        assert gym_env_adapter.env != new_env_mock
        assert gym_env_adapter.env_name == env_name

        gym_env_adapter.reset()
        gym_env_mock_obj.close.assert_not_called()
        assert gym_env_adapter.env == gym_env_mock_obj
        assert gym_env_adapter.env_name == env_name
