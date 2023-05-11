import numpy as np

from typing import Dict

from USTC_lab.data import LoggerFactory


class Status:
    """
    we calculate latest 100 episode, for XX rate such as reach rate and collision rate.
    """
    int_size = 100

    def __init__(self,
                 agent_num: int,
                 model_dtype: np.dtype,
                 logger_f: LoggerFactory,
                 start_score=0):
        
        self.agent_num = agent_num
        self.model_dtype = model_dtype
        self.logger_f = logger_f
        # for rewards logger / episode
        self.rewards_sum = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.rewards_episode = np.array([start_score] * self.agent_num, dtype=self.model_dtype)
        # for speeds logger / episode
        self.v_speeds_sum = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.v_speeds_episode = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.w_speeds_sum = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.w_speeds_episode = np.zeros(self.agent_num, dtype=self.model_dtype)
        # for speeds when the robot get close to human.
        self.v_speeds_sum_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.v_speeds_episode_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.w_speeds_sum_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.w_speeds_episode_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        # for speeds when the robot get away of human.
        self.v_speeds_sum_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.v_speeds_episode_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.w_speeds_sum_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.w_speeds_episode_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)

        # for rate logger / latest 100[int_size] episode
        self.episode_num = np.zeros(self.agent_num, dtype=np.int32)
        self.arrive_num = np.zeros([self.int_size, self.agent_num], dtype=self.model_dtype)
        self.coll_num = np.zeros([self.int_size, self.agent_num], dtype=self.model_dtype)
        self.coll_static_num = np.zeros([self.int_size, self.agent_num], dtype=self.model_dtype)
        self.coll_ped_num = np.zeros([self.int_size, self.agent_num], dtype=self.model_dtype)
        self.coll_robot_num = np.zeros([self.int_size, self.agent_num], dtype=self.model_dtype)

        # steps for each episode
        self.steps_sum = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.steps_episode = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.steps_sum_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.steps_episode_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.steps_sum_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.steps_episode_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        # tmp use tensor
        self.tmp = np.array(range(self.agent_num), dtype=np.int32)
        self.avg_v = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.avg_w = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.avg_v_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.avg_w_close_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.avg_v_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)
        self.avg_w_no_human = np.zeros(self.agent_num, dtype=self.model_dtype)

    def update_nav_status(self, info: Dict, dones: np.ndarray):
        all_down = info['all_down']
        self.arrive_num[self.episode_num % self.int_size, self.tmp] = info['arrive']
        self.coll_static_num[self.episode_num % self.int_size, self.tmp] = np.where(info['collision']==1, 1.0, 0)
        self.coll_ped_num[self.episode_num % self.int_size, self.tmp] = np.where(info['collision']==2, 1.0, 0)
        self.coll_robot_num[self.episode_num % self.int_size, self.tmp] = np.where(info['collision']==3, 1.0, 0)
        self.episode_num += np.where(all_down, 1, 0) # used for epi index of each agent,

        self.v_speeds_sum += abs(info.get('speeds', np.zeros([self.agent_num, 2], dtype=self.model_dtype))[:, 0]) 
        # record the latest episode speed sum
        self.v_speeds_episode = self.v_speeds_episode * (1 - all_down) + self.v_speeds_sum * all_down
        self.v_speeds_sum *= (1 - all_down)
        self.w_speeds_sum += abs(info.get('speeds', np.zeros([self.agent_num, 2], dtype=self.model_dtype))[:, 1])
        # record the latest episode speed sum
        self.w_speeds_episode = self.w_speeds_episode * (1 - all_down) + self.w_speeds_sum * all_down
        self.w_speeds_sum *= (1 - all_down)

        if info.get("bool_get_close_to_human") is not None:
            self.v_speeds_sum_close_human += abs(info.get('speeds', np.zeros([self.agent_num, 2], dtype=self.model_dtype))[:, 0]) * info["bool_get_close_to_human"]
            self.v_speeds_episode_close_human = self.v_speeds_episode_close_human * (1 - all_down) + self.v_speeds_sum_close_human * all_down
            self.v_speeds_sum_close_human *= (1 - all_down)
            self.w_speeds_sum_close_human += abs(info.get('speeds', np.zeros([self.agent_num, 2], dtype=self.model_dtype))[:, 1]) * info["bool_get_close_to_human"]
            self.w_speeds_episode_close_human = self.w_speeds_episode_close_human * (1 - all_down) + self.w_speeds_sum_close_human * all_down
            self.w_speeds_sum_close_human *= (1 - all_down)
            self.steps_sum_close_human += info['is_clean'] * info["bool_get_close_to_human"]
            self.steps_episode_close_human = self.steps_episode_close_human * (1 - all_down) + self.steps_sum_close_human * all_down
            self.steps_sum_close_human *= (1 - all_down)

            # open area ( no human )
            self.v_speeds_sum_no_human += abs(
                info.get('speeds', np.zeros([self.agent_num, 2], dtype=self.model_dtype))[:, 0]) * (
                                             1 - info.get("bool_get_close_to_human", 0))
            # record the latest episode speed sum
            self.v_speeds_episode_no_human = self.v_speeds_episode_no_human * (1 - all_down) + self.v_speeds_sum_no_human * all_down
            self.v_speeds_sum_no_human *= (1 - all_down)
            self.w_speeds_sum_no_human += abs(
                info.get('speeds', np.zeros([self.agent_num, 2], dtype=self.model_dtype))[:, 1]) * (
                                             1 - info.get("bool_get_close_to_human", 0))
            # record the latest episode speed sum
            self.w_speeds_episode_no_human = self.w_speeds_episode_no_human * (1 - all_down) + self.w_speeds_sum_no_human * all_down
            self.w_speeds_sum_no_human *= (1 - all_down)

            self.steps_sum_no_human += info['is_clean'] * (
                        1 - info.get("bool_get_close_to_human", np.zeros(self.agent_num, dtype=self.model_dtype)))
            self.steps_episode_no_human = self.steps_episode_no_human * (
                        1 - all_down) + self.steps_sum_no_human * all_down
            self.steps_sum_no_human *= (1 - all_down)

        self.steps_sum += info['is_clean']
        self.steps_episode = self.steps_episode * (1 - all_down) + self.steps_sum * all_down
        self.steps_sum *= (1 - all_down)

    def update_reward_status(self, dones: np.ndarray, rewards: np.ndarray):
        # extrinsic rewards , from environment
        self.rewards_sum += rewards
        # record the latest episode reward sum
        self.rewards_episode = self.rewards_episode * (1 - dones) + self.rewards_sum * dones
        self.rewards_sum *= (1 - dones)
        
    def update_collision_logger(self, steps: int, int_process_env_id: int, config_env: dict):
        # collision rate log to tensorboard
        # static obs
        static_collision_rate = np.sum(self.coll_static_num, 0) / np.clip(self.episode_num+1, 1, self.int_size)
        static_collision_rate_dict = {
            str(int_process_env_id * config_env["batch_num_per_env"] + i):
                np.mean(static_collision_rate[i*config_env["agent_num_per_env"]: (i+1)*config_env["agent_num_per_env"]]) for i in range(config_env["batch_num_per_env"])
            }
        self.logger_f.add((static_collision_rate_dict, steps), "StaticObsCollisionRateEpisode")
        # ped
        if config_env['ped_sim']['total'] > 0:
            ped_collision_rate = np.sum(self.coll_ped_num, 0) / np.clip(self.episode_num+1, 1, self.int_size)
            ped_collision_rate_dict = {
                str(int_process_env_id * config_env["batch_num_per_env"] + i):
                    np.mean(ped_collision_rate[i*config_env["agent_num_per_env"]: (i+1)*config_env["agent_num_per_env"]]) for i in range(config_env["batch_num_per_env"])}
            self.logger_f.add((ped_collision_rate_dict, steps), "PedCollisionRateEpisode")
        # other robot
        if config_env['agent_num_per_env'] > 1:
            other_robot_collision_rate = np.sum(self.coll_robot_num, 0) / np.clip(self.episode_num+1, 1, self.int_size)
            other_robot_collision_rate_dict = {
                str(int_process_env_id * config_env["batch_num_per_env"] + i):
                    np.mean(other_robot_collision_rate[i*config_env["agent_num_per_env"]: (i+1)*config_env["agent_num_per_env"]]) for i in range(config_env["batch_num_per_env"])}
            self.logger_f.add((other_robot_collision_rate_dict, steps), "OtherRobotCollisionRateEpisode")

    def update_reach_logger(self, steps: int, int_process_env_id: int, config_env: dict):
        # reach rate log to tensorboard
        arrive_rate = np.sum(self.arrive_num, 0) / np.clip(self.episode_num+1, 1, self.int_size)
        reach_rate_dict = {str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(arrive_rate[i*config_env["agent_num_per_env"]: (i+1)*config_env["agent_num_per_env"]])
                           for i in range(config_env["batch_num_per_env"])}
        self.logger_f.add((reach_rate_dict, steps), "ReachRateEpisode")

    def update_reward_logger(self, steps: int, int_process_env_id: int, config_env: dict):
        # reward log to tensorboard
        self.rewards_episode.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        r_dict = {str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.rewards_episode[i]) for
                  i in
                  range(config_env["batch_num_per_env"])}
        self.logger_f.add((r_dict, steps), "RewardEpisode")

    def update_action_speed_logger(self, steps: int, int_process_env_id: int, config_env: dict):
        """
            open area
        """
        self.avg_v = self.v_speeds_episode / self.steps_episode
        self.avg_w = self.w_speeds_episode / self.steps_episode

        self.avg_v.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        self.avg_w.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        v_speed_dict = {str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.avg_v[i]) for
                        i in
                        range(config_env["batch_num_per_env"])}
        w_speed_dict = {str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.avg_w[i]) for
                        i in
                        range(config_env["batch_num_per_env"])}
        self.logger_f.add((v_speed_dict, steps), "LinearVelocityEpisodeLogger")
        self.logger_f.add((w_speed_dict, steps), "AngularVelocityEpisodeLogger")

    def _update_action_speed_logger_close_human(self, steps: int, int_process_env_id: int, config_env: dict):
        """
                    avg velocity in the area which close to human
                """

        self.avg_v_close_human = np.where(self.steps_episode_close_human > 0,
                                          self.v_speeds_episode_close_human / self.steps_episode_close_human,
                                          self.avg_v_close_human)
        self.avg_w_close_human = np.where(self.w_speeds_episode_close_human > 0,
                                          self.w_speeds_episode_close_human / self.steps_episode_close_human,
                                          self.avg_w_close_human)

        self.avg_v_close_human.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        self.avg_w_close_human.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        v_speed_dict = {
            str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.avg_v_close_human[i]) for
            i in
            range(config_env["batch_num_per_env"])}
        w_speed_dict = {
            str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.avg_w_close_human[i]) for
            i in
            range(config_env["batch_num_per_env"])}
        self.logger_f.add((v_speed_dict, steps), "CloseHumanLinearVelocityEpisodeLogger")
        self.logger_f.add((w_speed_dict, steps), "CloseHumanAngularVelocityEpisodeLogger")

    def _update_action_speed_logger_no_human(self, steps: int, int_process_env_id: int, config_env: dict):
        """
                    no human area, but may still exist obstacles
        """
        # print(self.steps_episode_no_human, self.avg_v, self.v_speeds_episode, flush=True)
        self.avg_v_no_human = np.where(self.steps_episode_no_human > 0, self.v_speeds_episode_no_human / self.steps_episode_no_human,
                              self.avg_v_no_human)
        self.avg_w_no_human = np.where(self.steps_episode_no_human > 0, self.w_speeds_episode_no_human / self.steps_episode_no_human,
                              self.avg_w_no_human)

        self.avg_v_no_human.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        self.avg_w_no_human.reshape(config_env["batch_num_per_env"], config_env["agent_num_per_env"])
        v_speed_dict = {str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.avg_v_no_human[i]) for
                        i in
                        range(config_env["batch_num_per_env"])}
        w_speed_dict = {str(int_process_env_id * config_env["batch_num_per_env"] + i): np.mean(self.avg_w_no_human[i]) for
                        i in
                        range(config_env["batch_num_per_env"])}
        self.logger_f.add((v_speed_dict, steps), "OpenAreaLinearVelocityEpisodeLogger")
        self.logger_f.add((w_speed_dict, steps), "OpenAreaAngularVelocityEpisodeLogger")

    def update_ped_relation_velocity_logger(self, steps: int, int_process_env_id: int, config_env: dict):
        self._update_action_speed_logger_close_human(steps, int_process_env_id, config_env)
        self._update_action_speed_logger_no_human(steps, int_process_env_id, config_env)

    def group_logger(self, config_env):
        dict_group_logger = {"ReducedRewardLogger": np.mean(self.rewards_episode)}
        # add XXX rate
        """
        I simply use mean to aggregate rate logger.
        In the beginning of the training, it's not right, because total episodes is less than 100.
        """
        if config_env.get("env_type") == "robot_nav":
            dict_group_logger["ReducedReachRateLogger"] = np.mean(np.sum(self.arrive_num, 0) / np.clip(self.episode_num+1, 1, self.int_size))
            dict_group_logger["ReducedStaticObsCollisionRateLogger"] = np.mean(np.sum(self.coll_static_num, 0) / np.clip(self.episode_num+1, 1, self.int_size))
            if config_env['ped_sim']['total'] > 0:
                dict_group_logger["ReducedPedCollisionRateLogger"] = np.mean(np.sum(self.coll_ped_num, 0) / np.clip(self.episode_num+1, 1, self.int_size))
                dict_group_logger['ReducedCloseHumanLinearVelocityEpisodeLogger'] = np.mean(self.v_speeds_episode_close_human / self.steps_episode_close_human)
                dict_group_logger['ReducedCloseHumanAngularVelocityEpisodeLogger'] = np.mean(self.w_speeds_episode_close_human / self.steps_episode_close_human)
                dict_group_logger['ReducedOpenAreaLinearVelocityEpisodeLogger'] = np.mean(self.v_speeds_episode_no_human / self.steps_episode_no_human)
                dict_group_logger['ReducedOpenAreaAngularVelocityEpisodeLogger'] = np.mean(self.w_speeds_episode_no_human / self.steps_episode_no_human)
            if config_env['agent_num_per_env'] > 1:
                dict_group_logger["ReducedOtherRobotCollisionRateLogger"] = np.mean(np.sum(self.coll_robot_num, 0) / np.clip(self.episode_num+1, 1, self.int_size))

            dict_group_logger['ReducedLinearVelocityEpisodeLogger'] = np.mean(self.v_speeds_episode / self.steps_episode)
            dict_group_logger['ReducedAngularVelocityEpisodeLogger'] = np.mean(self.w_speeds_episode / self.steps_episode)
            dict_group_logger['ReducedStepsEpisodeLogger'] = np.mean(self.steps_episode)
        return dict_group_logger