#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
import math
from ppo.config import GameConfig
from collections import deque

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self.m_last_frame_totalHurtToHero =-1
        self.m_last_frame_totalBeHurtByHero =-1
        self.m_last_frame_soldier_av_hp = -1
        self.main_soldiers = []
        self.enemy_soldiers = []
        self.m_last_frame_hp = -1 # added
        self.m_last_frame_grass_status = False # added
        self.m_last_frame_received_enemy_hurt = -1 # added
        self.grass_position_list = [] # added
        self.m_last_frame_pos = [] #add
        self.m_last_frame_target = None
        self.last_few_frame_hp = deque(maxlen=5)

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        self.last_frame_data_process(frame_data)

        frame_no = frame_data["frameNo"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    # main_hero, enemy_hero = None, None
    # main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        #global main_hero, enemy_hero, main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers, enemy_soldiers

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]
        
        # Get both defense towers
        # 获取双方防御塔和小兵
        main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
                elif organ_subtype == "ACTOR_SUB_SOLDIER":
                    self.main_soldiers.append(organ)
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ
                elif organ_subtype == "ACTOR_SUB_SOLDIER":
                    self.enemy_soldiers.append(organ)
        #print(main_tower)
        hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
        tower_hit_tar_info =  main_tower['attack_target']#.get("hit_target_info", None)
 

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            
            #对英雄输出
            elif reward_name == "HurtToHero":
                distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
                #balance_rate 用来鼓励优先攻击英雄，当敌人处于攻击范围时
                hp_diff = main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]-enemy_hero["actor_state"]["hp"]/enemy_hero["actor_state"]["max_hp"]
                balance_rate = 1
                if distance_to_enemy <= main_hero['actor_state']['attack_range']:
                    balance_rate = 1.2
                #血量优劣势调整奖励
                if hp_diff>0.05:
                    balance_rate *=(1+hp_diff)
                if hp_diff<-0.35:
                    balance_rate*=hp_diff
                #草丛
                if main_hero["isInGrass"] is True:
                    balance_rate+=0.1
                #走位
                if self.m_last_frame_pos != (main_hero["actor_state"]["location"]["x"],main_hero["actor_state"]["location"]["z"]):
                    balance_rate+=0.1
                reward_struct.cur_frame_value = (main_hero["totalHurtToHero"]-self.m_last_frame_totalHurtToHero)*balance_rate/main_hero_max_hp
            
            elif reward_name == "HurtToOthers":
                balance_rate = 1
                if hit_target_info is not None and 'conti_hit_count' in hit_target_info[0]:
                    balance_rate+=0.03*hit_target_info['conti_hit_count']
                reward_struct.cur_frame_value = (main_hero["totalHurt"]-main_hero["totalHurtToHero"])/10000*balance_rate

            #承受英雄伤害
            elif reward_name == "BeHurtByHero":
                reward_struct.cur_frame_value = (main_hero["totalBeHurtByHero"]-self.m_last_frame_totalBeHurtByHero)/main_hero["actor_state"]["max_hp"]
            # elif reward_name == "HurtToHero":
            #     reward_struct.cur_frame_value = (main_hero["totalHurt"] - main_hero["totalHurtToHero"])
            #获取敌方小兵血量，平均，normailze by max
            elif reward_name == "enemy_Soldiers_hp":
                balance_rate=1
                total_hp = 0
                average_hp = 0
                if enemy_soldiers == [] or self.m_last_frame_soldier_av_hp == 0:
                    reward_struct.cur_frame_value = 0
                else:
                # hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
                # tower_hit_tar_info =  main_tower["actor_state"].get("hit_target_info", None)
                    if_hit_solder = 0
                    if_main_hit_solder = 0
                    for soldier in enemy_soldiers:
                        if soldier['runtime_id'] ==tower_hit_tar_info[0]['hit_target']:
                            if_main_hit_solder +=1
                        if soldier['runtime_id'] ==hit_target_info[0]['hit_target']:
                            if_hit_solder +=1

                    if  if_hit_solder==0 :
                        balance_rate = 0
                    
                    elif  enemy_soldiers != [] and self.main_soldiers != []:
                        #nearest_dist_enemy_soldier_to_hero = self.calculate_distance(main_hero["actor_state"]["location"],enemy_soldiers[0]["location"])
                        # if nearest_dist_enemy_soldier_to_hero <= main_hero['actor_state']['attack_range']:
                        #     balance_rate = 1
                        furthest_main_soldier = max(self.main_soldiers, key=lambda s: s['location']['z'])
                        max_distance=-1
                        for soldier in enemy_soldiers:
                            distance = self.calculate_distance(furthest_main_soldier['location'], soldier ['location'])
                            if distance > max_distance:
                                max_distance = distance
                                furthest_enemy_soldier_runtime_id = soldier ['runtime_id']
                        #hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
                        
                        if if_hit_solder!=0:
                            balance_rate+=0.2
                        #优先攻击后排小兵
                        if hit_target_info is not None and hit_target_info == furthest_enemy_soldier_runtime_id:
                            balance_rate+=0.1
                
                        
                        for soldier in enemy_soldiers:
                            total_hp+=soldier['hp']/soldier['max_hp']
                        average_hp = self.m_last_frame_soldier_av_hp-total_hp/len(enemy_soldiers)
                    reward_struct.cur_frame_value = average_hp*balance_rate
    
    
    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if self.main_soldiers != []:
            s_pos = (self.main_soldiers[0]["location"]["x"],self.main_soldiers[0]["location"]["x"])
            dist_s2emy = math.dist(s_pos,enemy_tower_pos)
            if self.enemy_soldiers ==[] and dist_hero2emy >8800 and dist_hero2emy >=dist_s2emy and dist_hero2emy < dist_main2emy:
                
                #鼓励朝塔前进
                forward_value += (dist_hero2emy-dist_s2emy)/dist_main2emy
            #鼓励守塔
        if self.enemy_soldiers!=[] and len(self.enemy_soldiers)>=2:
            es_pos = (self.enemy_soldiers[0]["location"]["x"],self.main_soldiers[0]["location"]["x"])
            dist_es2main = math.dist(es_pos,main_tower_pos)
            if dist_es2main<9000 and dist_hero2emy < dist_main2emy-9000:
                forward_value +=(dist_hero2emy - dist_main2emy ) / dist_main2emy

        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value += (dist_main2emy - dist_hero2emy) / dist_main2emy
        #不回撤
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] <0.15 and dist_hero2emy < dist_main2emy:
            forward_value += (dist_hero2emy - dist_main2emy ) *0.01/ dist_main2emy
        #大幅度减血鼓励撤退
        if self.check_decresing_hp(self.last_few_frame_hp):
            forward_value += (dist_hero2emy - dist_main2emy)/ dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)
    
###########################################################
    def last_frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.m_last_frame_hp = hero["actor_state"]["hp"]
                if hero["actor_state"].get("hit_target_info", None) is not None:
                    self.m_last_frame_target = hero["actor_state"].get("hit_target_info", None)[0]['hit_target']  
                else:
                    self.m_last_frame_target = None
                self.m_last_frame_pos = (hero["actor_state"]["location"]["x"],hero["actor_state"]["location"]["z"])
                self.m_last_frame_grass_status = hero["isInGrass"]
                self.m_last_frame_received_enemy_hurt = hero["totalBeHurtByHero"]
                self.main_hero_camp = main_camp
                self.m_last_frame_totalHurtToHero =hero["totalHurtToHero"]
                self.m_last_frame_totalBeHurtByHero =hero["totalBeHurtByHero"]
                self.last_few_frame_hp.append(hero["actor_state"]["hp"])
                if self.enemy_soldiers != []:
                    total_hp = 0
                    for soldier in self.enemy_soldiers:
                        total_hp+=soldier['hp']/soldier['max_hp']
                    self.m_last_frame_soldier_av_hp = total_hp/len(self.enemy_soldiers)
                else:self.m_last_frame_soldier_av_hp = 0
            else:
                enemy_camp = hero["actor_state"]["camp"]
#######################################################################

#################################################################
    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def calculate_distance(self,loc1, loc2):
        return math.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['z'] - loc2['z'])**2)

   
    
    # 判断是否连续多帧大幅度掉血
    def check_decresing_hp(self, hp_queue):
        # for i in range(1, len(hp_queue)):
        #     current = hp_queue[i]
        #     previous = hp_queue[i - 1]
            
        #     if current >= previous * 0.9:
        #         return False
        if len(hp_queue) >1 and hp_queue[len(hp_queue)] <= 0.4 * hp_queue[0]:
            return True
        return False

    
##################################################################  
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
     
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                balance_rate = 1
                if frame_data['frameNo']>8000:
                    balance_rate = 1.2
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value =(reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate
            
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
           
            elif reward_name == "kill":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                balance_rate = -1
                if frame_data['frameNo']>6000 or reward_struct.cur_frame_value >=2 :
                    balance_rate = 1
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate

            elif reward_name == "death":
                balance_rate = 1
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if frame_data['frameNo']>5000:
                    balance_rate *=1.00005**frame_data['frameNo']
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate
                


            elif reward_name == "HurtToHero":
                
                # distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
                # #balance_rate 用来鼓励优先攻击英雄，当敌人处于攻击范围时
                # hp_diff = main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]-enemy_hero["actor_state"]["hp"]/enemy_hero["actor_state"]["max_hp"]
                # balance_rate = 1
                # if distance_to_enemy <= main_hero['actor_state']['attack_range']:
                #     balance_rate = 1.2
                # #血量优劣势调整奖励
                # if hp_diff>0.05:
                #     balance_rate *=(1+hp_diff)
                # if hp_diff<-0.35:
                #     balance_rate*=hp_diff
                # #草丛
                # if main_hero["isInGrass"] is True:
                #     balance_rate+=0.1
                # #走位
                # if self.m_last_frame_pos != (main_hero["actor_state"]["location"]["x"],main_hero["actor_state"]["location"]["z"]):
                #     balance_rate+=0.1
                # distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"], enemy_hero["actor_state"]["location"])
    
                # # main_hero 视角的血量差异
                # hp_diff_main = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] - enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
                
                # balance_rate_main = 1
                # if distance_to_enemy <= main_hero['actor_state']['attack_range']:
                #     balance_rate_main = 1.2  # 鼓励近距离攻击敌方英雄

                # # 根据血量差异调整平衡率
                # if hp_diff_main > 0.05:
                #     balance_rate_main *= (1 + hp_diff_main)
                # elif hp_diff_main < -0.35:
                #     balance_rate_main *= hp_diff_main

                # # 鼓励草丛埋伏和走位
                # if main_hero["isInGrass"]:
                #     balance_rate_main += 0.1
                # if self.m_last_frame_pos != (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"]):
                #     balance_rate_main += 0.1

                # # 计算 enemy_hero 视角的平衡比率 balance_rate_enemy
                # hp_diff_enemy = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"] - main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
                
                # balance_rate_enemy = 1
                # if distance_to_enemy <= enemy_hero['actor_state']['attack_range']:
                #     balance_rate_enemy = 1.2  # 鼓励敌方近距离攻击

                # if hp_diff_enemy > 0.05:
                #     balance_rate_enemy *= (1 + hp_diff_enemy)
                # elif hp_diff_enemy < -0.35:
                #     balance_rate_enemy *= hp_diff_enemy

                # if enemy_hero["isInGrass"]:
                #     balance_rate_enemy += 0.1

                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                

            elif reward_name == "BeHurtByHero":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                #distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)
            
            elif reward_name == "HurtToOthers":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)
            
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value =-(reward_struct.cur_frame_value - reward_struct.last_frame_value) 

            
            elif reward_name == "enemy_Soldiers_hp":
                
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate
                #print(reward_struct.value)
            
                    
            
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum





# import math
# from ppo.config import GameConfig
# from collections import deque

# # Used to record various reward information
# # 用于记录各个奖励信息
# class RewardStruct:
#     def __init__(self, m_weight=0.0):
#         self.cur_frame_value = 0.0
#         self.last_frame_value = 0.0
#         self.value = 0.0
#         self.weight = m_weight
#         self.min_value = -1
#         self.is_first_arrive_center = True


# # Used to initialize various reward information
# # 用于初始化各个奖励信息
# def init_calc_frame_map():
#     calc_frame_map = {}
#     for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
#         calc_frame_map[key] = RewardStruct(weight)
#     return calc_frame_map


# class GameRewardManager:
#     def __init__(self, main_hero_runtime_id):
#         self.main_hero_player_id = main_hero_runtime_id
#         self.main_hero_camp = -1
#         self.main_hero_hp = -1
#         self.main_hero_organ_hp = -1
#         self.m_reward_value = {}
#         self.m_last_frame_no = -1
#         self.m_cur_calc_frame_map = init_calc_frame_map()
#         self.m_main_calc_frame_map = init_calc_frame_map()
#         self.m_enemy_calc_frame_map = init_calc_frame_map()
#         self.m_init_calc_frame_map = {}
#         self.time_scale_arg = GameConfig.TIME_SCALE_ARG
#         self.m_main_hero_config_id = -1
#         self.m_each_level_max_exp = {}
#         self.m_last_frame_hp = 0 # added
#         self.m_last_frame_grass_status = False # added
#         self.m_last_frame_received_enemy_hurt = 0 # added
#         self.grass_position_list = [] # added
#         self.m_last_frame_target = None

#         global last_few_frame_hp
#         last_few_frame_hp = deque(maxlen=5)

#     # Used to initialize the maximum experience value for each agent level
#     # 用于初始化智能体各个等级的最大经验值
#     def init_max_exp_of_each_hero(self):
#         self.m_each_level_max_exp.clear()
#         self.m_each_level_max_exp[1] = 160
#         self.m_each_level_max_exp[2] = 298
#         self.m_each_level_max_exp[3] = 446
#         self.m_each_level_max_exp[4] = 524
#         self.m_each_level_max_exp[5] = 613
#         self.m_each_level_max_exp[6] = 713
#         self.m_each_level_max_exp[7] = 825
#         self.m_each_level_max_exp[8] = 950
#         self.m_each_level_max_exp[9] = 1088
#         self.m_each_level_max_exp[10] = 1240
#         self.m_each_level_max_exp[11] = 1406
#         self.m_each_level_max_exp[12] = 1585
#         self.m_each_level_max_exp[13] = 1778
#         self.m_each_level_max_exp[14] = 1984

#     def result(self, frame_data):
#         self.init_max_exp_of_each_hero()
#         self.frame_data_process(frame_data)
#         self.get_reward(frame_data, self.m_reward_value)

#         frame_no = frame_data["frameNo"]
#         if self.time_scale_arg > 0:
#             for key in self.m_reward_value:
#                 self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

#         return self.m_reward_value

#     # Calculate the value of each reward item in each frame
#     # 计算每帧的每个奖励子项的信息
#     main_hero, enemy_hero = None, None
#     main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
#     def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
#         global main_hero, enemy_hero, main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers, enemy_soldiers

#         # Get both agents
#         # 获取双方智能体
#         main_hero, enemy_hero = None, None
#         hero_list = frame_data["hero_states"]
#         for hero in hero_list:
#             hero_camp = hero["actor_state"]["camp"]
#             if hero_camp == camp:
#                 main_hero = hero
#             else:
#                 enemy_hero = hero
#         main_hero_hp = main_hero["actor_state"]["hp"]
#         main_hero_max_hp = main_hero["actor_state"]["max_hp"]
#         main_hero_ep = main_hero["actor_state"]["values"]["ep"]
#         main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

#         # Get both defense towers
#         # 获取双方防御塔和小兵
#         main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
#         npc_list = frame_data["npc_states"]
#         for organ in npc_list:
#             organ_camp = organ["camp"]
#             organ_subtype = organ["sub_type"]
#             if organ_camp == camp:
#                 if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
#                     main_tower = organ
#                 elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
#                     main_spring = organ
#                 elif organ_subtype == "ACTOR_SUB_SOLDIER":
#                     main_soldiers.append(organ)
#             else:
#                 if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
#                     enemy_tower = organ
#                 elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
#                     enemy_spring = organ
#                 elif organ_subtype == "ACTOR_SUB_SOLDIER":
#                     enemy_soldiers.append(organ)
    
        
#         #print("\n","enemy_location:",enemy_hero["actor_state"]["location"])
#         #print("hero_location:",Main_camp_hero_state_common_feature,"\n")
        
#         #print("\n","npc_states:",frame_data["npc_states"],"\n")
#         #print("\n","main_soldiers:",main_soldiers,"\n")

#         # npc_data = set()  # 使用set来去重
#         # for npc in frame_data['npc_states']:
#         #     npc_info = (npc['config_id'], npc['runtime_id'], npc['actor_type'], npc['sub_type'])  # 使用元组
#         #     if npc_info not in npc_data:
#         #         npc_data.add(npc_info)
#         #         print(f"New NPC added: {npc_info}")
#         #         print(f"Current NPC data set: {npc_data}")


 
#         # hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
#         # if hit_target_info is not None:
#         #     print("\n", "hit_target_info:", hit_target_info, "\n")
#         # else:
#         #     print("hit_target_info not found in frame_data.")
        
#         # real_cmd = main_hero.get("real_cmd",None)
#         # if real_cmd is not None:
#         #     print("\n", "real_cmd:", real_cmd, "\n")
#         # else:
#         #     print("real_cmd not found in frame_data.")



    

#         for reward_name, reward_struct in cul_calc_frame_map.items():
#             reward_struct.last_frame_value = reward_struct.cur_frame_value
#             # Money
#             # 金钱
#             if reward_name == "money":
#                 reward_struct.cur_frame_value = main_hero["moneyCnt"]
#             # Health points
#             # 生命值
#             elif reward_name == "hp_point":
#                 reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
#             # Energy points
#             # 法力值
#             elif reward_name == "ep_rate":
#                 if main_hero_max_ep == 0 or main_hero_hp <= 0:
#                     reward_struct.cur_frame_value = 0
#                 else:
#                     reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
#             # Kills
#             # 击杀
#             elif reward_name == "kill":
#                 reward_struct.cur_frame_value = main_hero["killCnt"]
#             # Deaths
#             # 死亡
#             elif reward_name == "death":
#                 reward_struct.cur_frame_value = main_hero["deadCnt"]
#             # Tower health points
#             # 塔血量
#             elif reward_name == "tower_hp_point":
#                 reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
#             # Last hit
#             # 补刀
#             elif reward_name == "last_hit":
#                 reward_struct.cur_frame_value = 0.0
#                 frame_action = frame_data["frame_action"]
#                 if "dead_action" in frame_action:
#                     dead_actions = frame_action["dead_action"]
#                     for dead_action in dead_actions:
#                         if (
#                             dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
#                             and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
#                         ):
#                             reward_struct.cur_frame_value += 1.0
#                         elif (
#                             dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
#                             and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
#                         ):
#                             reward_struct.cur_frame_value -= 1.0
#             # Experience points
#             # 经验值
#             elif reward_name == "exp":
#                 reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
#             # Forward
#             # 前进
#             elif reward_name == "forward":
#                 reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            
#             #对英雄输出
#             elif reward_name == "HurtToHero":
#                 reward_struct.cur_frame_value = main_hero["totalHurtToHero"]
            
#             elif reward_name == "HurtToOthers":
#                 reward_struct.cur_frame_value = main_hero["totalHurt"]-main_hero["totalHurtToHero"]

#             #承受英雄伤害
#             elif reward_name == "BeHurtByHero":
#                 reward_struct.cur_frame_value = main_hero["totalBeHurtByHero"]
#             # elif reward_name == "HurtToHero":
#             #     reward_struct.cur_frame_value = (main_hero["totalHurt"] - main_hero["totalHurtToHero"])
#             #获取敌方小兵血量，平均，normailze by max
#             elif reward_name == "enemy_Soldiers_hp":
#                 total_hp = 0
#                 if enemy_soldiers !=[]:
#                     for soldier in enemy_soldiers:
#                         total_hp+=soldier['hp']/soldier['max_hp']
#                     average_hp = total_hp/len(enemy_soldiers)
#                     reward_struct.cur_frame_value = average_hp
#                 else:
#                     reward_struct.cur_frame_value = 0
#             # Hurt to enemy
#     # 对英雄伤害
#             elif reward_name == "attack_enemy":
#                 if self.calculate_attack_enemy_in_grass_reward(main_hero, enemy_hero) != 0:
#                     reward_struct.cur_frame_value = 0
#                 reward_struct.cur_frame_value = self.calculate_attack_enemy_reward(enemy_hero)
#             # Received hurt
#             # 受到的伤害
#             elif reward_name == "received_hurt":
#                 reward_struct.cur_frame_value = -self.calculate_received_hurt_reward(main_hero)

#             elif reward_name == "hide_grass":
#                 reward_struct.cur_frame_value = self.calculate_hide_in_grass_reward(main_hero)
#             # Attack enemy when in grass
#             # 在草里攻击地方英雄
#             elif reward_name == "hide_grass_attack":
#                 reward_struct.cur_frame_value = self.calculate_attack_enemy_in_grass_reward(main_hero, enemy_hero)
#             # Received hurt then hide into grass
#             # 受到攻击躲进草里
#             elif reward_name == "hide_grass_hurt":
#                 reward_struct.cur_frame_value = self.calculate_hide_in_grass_when_received_hurt(main_hero)
#             # Fix target
#             # 固定目标
#             elif reward_name == "fix_target":
#                 reward_struct.cur_frame_value = self.calculate_fix_target_reward(main_hero)
#             # Run when cannot win
#             # 打不过就跑
#             elif reward_name == "run_when_cannot_win":
#                 reward_struct.cur_frame_value = self.calculate_run_when_cannot_win(main_hero,enemy_hero)
    
#     '''
#     这里注意一下，可以暂时放弃
#     '''
#     def calculate_attack_enemy_reward(self, enemy_hero):
#             if enemy_hero is None:
#                 return 0.0
#             if enemy_hero["actor_state"]["hp"] == 0:
#                 enemy_hero["actor_state"]["hp"] = 1
#             hurt_to_enemy_value = enemy_hero["totalBeHurtByHero"] / enemy_hero["actor_state"]["hp"]
#             return hurt_to_enemy_value

#         # Calcule the reward based on the received hurt and max hp (DONE)
#         # 用智能体受到的伤害和最大生命值，计算受到伤害的奖励 (DONE)
#     def calculate_received_hurt_reward(self, main_hero):
#         received_hurt = main_hero["actor_state"]["hp"] - self.m_last_frame_hp
#         received_hurt_value = received_hurt / main_hero["actor_state"]["max_hp"]
#         return received_hurt_value

#     # Calculate the reward based on agent fix target (DONE)
#     # 智能体锁定目标奖励 (DONE)
#     def calculate_fix_target_reward(self,main_hero):
#         if main_hero["actor_state"].get("hit_target_info", None) is None:
#             return 0.0
#         current_target = main_hero["actor_state"].get("hit_target_info", None)[0]['hit_target']
#         if current_target == self.m_last_frame_target:
#             return 1.0
#         else:
#             return -1.0
    
#     # 判断是否连续多帧大幅度掉血
#     def check_decresing_hp(self, hp_queue):
#         for i in range(1, len(hp_queue)):
#             current = hp_queue[i]
#             previous = hp_queue[i - 1]
            
#             if current >= previous * 0.9:
#                 return False
            
#         if hp_queue[-1] <= 0.2 * hp_queue[0]:
#             return True
#         return False

#     # Calculate the reward if the agent find it cannot win the battle at this moment (DONE)
#     # 如果智能体发现打不过，跑路，并给奖励 (DONE)
#     def calculate_run_when_cannot_win(self, main_hero, enemy_hero):
#         if enemy_hero is None:
#             return 0.0

#         if main_hero["totalHurtToHero"] == 0:
#             main_hero_hurt_attack_rate = main_hero["totalBeHurtByHero"] / 1
#         else:
#             main_hero_hurt_attack_rate = main_hero["totalBeHurtByHero"] / main_hero["totalHurtToHero"]
#         if enemy_hero["totalHurtToHero"] == 0:
#              enemy_hero_hurt_attack_rate = enemy_hero["totalBeHurtByHero"] / 1
#         else:
#             enemy_hero_hurt_attack_rate = enemy_hero["totalBeHurtByHero"] / enemy_hero["totalHurtToHero"]
        
#         if main_hero_hurt_attack_rate >= enemy_hero_hurt_attack_rate:
#             """
#             锁定目标，继续攻击
#             """

#             if not self.check_decresing_hp(last_few_frame_hp):
#                 return self.calculate_fix_target_reward(main_hero)
#             current_target = main_hero["actor_state"].get("hit_target_info", None)[0]['hit_target']
#             if current_target == self.m_last_frame_target:
#                 return -1.0
#             return 1.0
#         else:
#             """
#             跑路
#             """
#             if self.check_decresing_hp(last_few_frame_hp):
#                 current_target = main_hero["actor_state"].get("hit_target_info", None)[0]['hit_target']
#                 if current_target == self.m_last_frame_target:
#                     return -1.0
#                 return 1.0
#             return 0.0
#     """
#     Add new rewards based on grasses
#     新增基于草丛计算的奖励

#     Need to add more detail in result & get_reward
#     需要在result和get_reward中加入更多内容
#     """
    
#     # Calculate the reward based on entering grass, and storing the grass position (DONE)
#     # 用智能体是否进入草丛，计算进入草丛的奖励，并储存草丛位置 (DONE)
#     def calculate_hide_in_grass_reward(self, main_hero):
#         if main_hero["isInGrass"] is True:
#             if main_hero["actor_state"]["location"] not in self.grass_position_list:
#                 self.grass_position_list.append(main_hero["actor_state"]["location"])
                
#         if self.m_last_frame_grass_status is False:
#             if main_hero["isInGrass"] is True:
#                 return 1.0
#         return 0.0

#     # Calculate the reward based on attack enermy when in the grass (DONE)
#     # 用智能体是否在草丛内攻击敌人，计算攻击敌人的奖励 (DONE)
#     def calculate_attack_enemy_in_grass_reward(self, main_hero, enemy_hero):
#         if main_hero["isInGrass"] is True:
#             hurt_to_enemy_value = self.calculate_attack_enemy_reward(enemy_hero=enemy_hero)
#             return hurt_to_enemy_value
#         return 0.0
#     def calculate_hide_in_grass_when_received_hurt(self, main_hero):
#         if self.m_last_frame_received_enemy_hurt < main_hero["totalBeHurtByHero"]:
#             return self.calculate_hide_in_grass_reward(main_hero=main_hero)
#         return 0.0



#     # Calculate the total amount of experience gained using agent level and current experience value
#     # 用智能体等级和当前经验值，计算获得经验值的总量
#     def calculate_exp_sum(self, this_hero_info):
#         exp_sum = 0.0
#         for i in range(1, this_hero_info["level"]):
#             exp_sum += self.m_each_level_max_exp[i]
#         exp_sum += this_hero_info["exp"]
#         return exp_sum

#     # Calculate the forward reward based on the distance between the agent and both defensive towers
#     # 用智能体到双方防御塔的距离，计算前进奖励
#     def calculate_forward(self, main_hero, main_tower, enemy_tower):
#         main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
#         enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
#         hero_pos = (
#             main_hero["actor_state"]["location"]["x"],
#             main_hero["actor_state"]["location"]["z"],
#         )
#         dist_s2emy =100000000000000000
#         forward_value = 0
#         dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
#         dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
#         if main_soldiers != []:
#             s_pos = (main_soldiers[0]["location"]["x"],main_soldiers[0]["location"]["x"])
#             dist_s2emy = math.dist(s_pos,enemy_tower_pos)
#         if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
#             forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
#         if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] <0.15:
#             forward_value = -1000/dist_hero2emy
#         if dist_s2emy<10000 and dist_hero2emy>dist_s2emy:
#             forward_value +=math.exp(-0.00001*dist_hero2emy)
#         return forward_value

#     # Calculate the reward item information for both sides using frame data
#     # 用帧数据来计算两边的奖励子项信息
#     def frame_data_process(self, frame_data):
#         main_camp, enemy_camp = -1, -1
#         for hero in frame_data["hero_states"]:
#             if hero["player_id"] == self.main_hero_player_id:
#                 main_camp = hero["actor_state"]["camp"]
#                 self.m_last_frame_hp = hero["actor_state"]["hp"]
#                 if hero["actor_state"].get("hit_target_info", None) is not None:
#                     self.m_last_frame_target = hero["actor_state"].get("hit_target_info", None)[0]['hit_target']
#                 else:
#                     self.m_last_frame_target = None
#                 self.m_last_frame_grass_status = hero["isInGrass"]
#                 self.m_last_frame_received_enemy_hurt = hero["totalBeHurtByHero"]
#                 self.main_hero_camp = main_camp
#                 last_few_frame_hp.append(hero["actor_state"]["hp"])
#             else:
#                 enemy_camp = hero["actor_state"]["camp"]
#         self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
#         self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

#     # Use the values obtained in each frame to calculate the corresponding reward value
#     # 用每一帧得到的奖励子项信息来计算对应的奖励值

#     def calculate_distance(self,loc1, loc2):
#         return math.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['z'] - loc2['z'])**2)

    
#     def get_reward(self, frame_data, reward_dict):
#         reward_dict.clear()
#         reward_sum, weight_sum = 0.0, 0.0
     
#         for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
#             if reward_name == "hp_point":
#                 balance_rate = 1
#                 if frame_data['frameNo']>8000:
#                     balance_rate = 2
#                 if (
#                     self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
#                     and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
#                 ):
#                     reward_struct.cur_frame_value = 0
#                     reward_struct.last_frame_value = 0
#                 elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
#                     reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
#                     reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
#                 elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
#                     reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
#                     reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
#                 else:
#                     reward_struct.cur_frame_value = (
#                         self.m_main_calc_frame_map[reward_name].cur_frame_value
#                         - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
#                     )
#                     reward_struct.last_frame_value = (
#                         self.m_main_calc_frame_map[reward_name].last_frame_value
#                         - self.m_enemy_calc_frame_map[reward_name].last_frame_value
#                     )
#                 reward_struct.value =(reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate
            
#             elif reward_name == "ep_rate":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 if reward_struct.last_frame_value > 0:
#                     reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
#                 else:
#                     reward_struct.value = 0
            
#             elif reward_name == "exp":
#                 main_hero = None
#                 for hero in frame_data["hero_states"]:
#                     if hero["player_id"] == self.main_hero_player_id:
#                         main_hero = hero
#                 if main_hero and main_hero["level"] >= 15:
#                     reward_struct.value = 0
#                 else:
#                     reward_struct.cur_frame_value = (
#                         self.m_main_calc_frame_map[reward_name].cur_frame_value
#                         - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
#                     )
#                     reward_struct.last_frame_value = (
#                         self.m_main_calc_frame_map[reward_name].last_frame_value
#                         - self.m_enemy_calc_frame_map[reward_name].last_frame_value
#                     )
#                     reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
#             elif reward_name == "forward":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "last_hit":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "kill":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 balance_rate = -1
#                 if frame_data['frameNo']>6000 or reward_struct.cur_frame_value >=2 :
#                     balance_rate = 1
#                 reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate

#             elif reward_name == "death":
#                 balance_rate = 1
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 if frame_data['frameNo']>5000:
#                     balance_rate *=1.00005**frame_data['frameNo']
#                 reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate
                



#             elif reward_name == "HurtToHero":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
#                 #balance_rate 用来鼓励优先攻击英雄，当敌人处于攻击范围时
#                 balance_rate = 1
#                 if distance_to_enemy <= main_hero['actor_state']['attack_range']:
#                     balance_rate = 1.2
#                 reward_struct.value = balance_rate*(reward_struct.cur_frame_value - reward_struct.last_frame_value)*math.exp(0.0005*distance_to_enemy)
                

#             elif reward_name == "BeHurtByHero":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
#                 reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)/min(3,math.exp(0.0008*distance_to_enemy))
            
#             elif reward_name == "HurtToOthers":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            
#             elif reward_name == "tower_hp_point":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
#                 reward_struct.value =-(reward_struct.cur_frame_value - reward_struct.last_frame_value) 

            
#             elif reward_name == "enemy_Soldiers_hp":
#                 reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#                 reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                
#                 #balance_rate 用来鼓励优先攻击英雄，当敌人处于攻击范围时
#                 balance_rate = 1.2
#                 bonus_rate = 1
#                 if reward_struct.last_frame_value==0 or -(reward_struct.cur_frame_value - reward_struct.last_frame_value)<150:
#                     bonus_rate=0

#                 if enemy_soldiers != [] and main_soldiers != []:
#                     nearest_dist_enemy_soldier_to_hero = self.calculate_distance(main_hero["actor_state"]["location"],enemy_soldiers[0]["location"])
#                     if nearest_dist_enemy_soldier_to_hero <= main_hero['actor_state']['attack_range']:
#                         balance_rate = 1
#                     furthest_main_soldier = max(main_soldiers, key=lambda s: s['location']['z'])
#                     max_distance=-1000000000
#                     for soldier in enemy_soldiers:
#                         distance = self.calculate_distance(furthest_main_soldier['location'], soldier ['location'])
#                         if distance > max_distance:
#                             max_distance = distance
#                             furthest_enemy_soldier_runtime_id = soldier ['runtime_id']
#                     hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
#                     #优先攻击后排小兵
#                     if hit_target_info is not None and hit_target_info == furthest_enemy_soldier_runtime_id:
#                         bonus_rate = 1.1
                
#                 reward_struct.value = -(reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate*bonus_rate
#                 #print(reward_struct.value)
            
#             elif reward_name == "attack_enemy":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "received_hurt":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "economy_advantage":
#                 if (
#                     self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
#                     and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
#                 ):
#                     reward_struct.cur_frame_value = 0
#                     reward_struct.last_frame_value = 0
#                 else:
#                     reward_struct.cur_frame_value = (
#                         self.m_main_calc_frame_map[reward_name].cur_frame_value
#                         - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
#                     )
#                     reward_struct.last_frame_value = (
#                         self.m_main_calc_frame_map[reward_name].last_frame_value
#                         - self.m_enemy_calc_frame_map[reward_name].last_frame_value
#                     )
#                 reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
#             elif reward_name == "hide_grass":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "hide_grass_attack":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "hide_grass_hurt":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "run_when_connot_win":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
#             elif reward_name == "fix_target":
#                 reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                    

            

#             else:
#                 reward_struct.cur_frame_value = (
#                     self.m_main_calc_frame_map[reward_name].cur_frame_value
#                     - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
#                 )
#                 reward_struct.last_frame_value = (
#                     self.m_main_calc_frame_map[reward_name].last_frame_value
#                     - self.m_enemy_calc_frame_map[reward_name].last_frame_value
#                 )
#                 reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

#             weight_sum += reward_struct.weight
#             reward_sum += reward_struct.value * reward_struct.weight
#             reward_dict[reward_name] = reward_struct.value
#         reward_dict["reward_sum"] = reward_sum
