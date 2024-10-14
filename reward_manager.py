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
        self.m_last_frame_totalHurtToHero =-1#add
        self.m_last_frame_totalBeHurtByHero =-1#add
        self.m_last_frame_soldier_av_hp = -1#add
        self.main_soldiers = []
        self.enemy_soldiers = []
        self.m_last_frame_hp = -1 # added
        self.m_last_frame_grass_status = False # added
        self.m_last_frame_received_enemy_hurt = -1 # added
        self.grass_position_list = [] # added
        self.m_last_frame_pos = [] #add
        self.m_last_frame_target = None
        self.last_few_frame_hp = deque(maxlen=8)
        

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
        self.m_last_frame_no = frame_no-1
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
        #print(self.last_few_frame_hp)
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        #print(main_hero_hp)
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]
        
        # Get both defense towers
        # 获取双方防御塔和小兵
        main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
        self.main_soldiers.clear()
        self.enemy_soldiers.clear()

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
        #print(main_hero['skill_state'])

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
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower,main_spring)
            
            elif reward_name == "heal":
                reward_struct.cur_frame_value = main_hero['skill_state']['slot_states'][4]['usedTimes']
            
            elif reward_name == "skill_hit_count":
                reward_struct.cur_frame_value = sum(main_hero['skill_state']['slot_states'][n]['hitHeroTimes'] for n in range(len(main_hero['skill_state']['slot_states'])))


            #对英雄输出
            elif reward_name == "HurtToHero":
                distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
                #balance_rate 用来鼓励优先攻击英雄，当敌人处于攻击范围时
                #hp_diff = main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]-enemy_hero["actor_state"]["hp"]/enemy_hero["actor_state"]["max_hp"]
                balance_rate = 1
                if distance_to_enemy <= main_hero['actor_state']['attack_range']:
                    balance_rate = 1.2
                #血量优劣势调整奖励
                # if hp_diff>0.05:
                #     balance_rate *=(1+hp_diff)
                # if hp_diff<-0.35:
                #     balance_rate*=hp_diff
                #草丛
                if main_hero["isInGrass"] is True:
                    balance_rate+=0.1
                #走位
                if self.m_last_frame_pos != (main_hero["actor_state"]["location"]["x"],main_hero["actor_state"]["location"]["z"]):
                    balance_rate+=0.1
                reward_struct.cur_frame_value = (main_hero["totalHurtToHero"]-self.m_last_frame_totalHurtToHero)*balance_rate/enemy_hero['actor_state']['max_hp']*math.sqrt(main_hero_hp/main_hero_max_hp)
            
            elif reward_name == "HurtToOthers":
                balance_rate = 1
                if hit_target_info is not None and 'conti_hit_count' in hit_target_info[0]:
                    #print(max([item['conti_hit_count'] for item in hit_target_info if 'conti_hit_count' in item]))
                    balance_rate+=0.03*max([item['conti_hit_count'] for item in hit_target_info if 'conti_hit_count' in item])
                reward_struct.cur_frame_value = (main_hero["totalHurt"]-main_hero["totalHurtToHero"])/100000*balance_rate*math.sqrt(main_hero_hp/main_hero_max_hp)

            #承受英雄伤害
            elif reward_name == "BeHurtByHero":
                reward_struct.cur_frame_value = (main_hero["totalBeHurtByHero"]-self.m_last_frame_totalBeHurtByHero)/main_hero["actor_state"]["max_hp"]*math.exp(-math.sqrt(main_hero_ep/main_hero_max_hp)/2)
            
            elif reward_name == "enemy_Soldiers_hp":
                balance_rate=1
                total_hp = 0
                average_hp = 0
                if self.enemy_soldiers == []:
                    reward_struct.cur_frame_value = 0
                else:
                # hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
                # tower_hit_tar_info =  main_tower["actor_state"].get("hit_target_info", None)
                    if_hit_solder = 0
                    if_main_hit_solder = 0
                    for soldier in self.enemy_soldiers:
                        if soldier['runtime_id'] ==tower_hit_tar_info: #['hit_target']:
                            if_main_hit_solder +=1
                        if hit_target_info != None and soldier['runtime_id'] ==hit_target_info[0]['hit_target']:
                            if_hit_solder +=1

                    if  if_hit_solder!=0 :
                        balance_rate *=1.05
                    
                    if self.main_soldiers != []:
                       
                        furthest_main_soldier = max(self.main_soldiers, key=lambda s: s['location']['z'])
                        max_distance=-1
                        for soldier in self.enemy_soldiers:
                            distance = self.calculate_distance(furthest_main_soldier['location'], soldier ['location'])
                            if distance > max_distance:
                                max_distance = distance
                                furthest_enemy_soldier_runtime_id = soldier ['runtime_id']
                        #hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
                        
                        if if_main_hit_solder!=0 and if_hit_solder !=0:
                            balance_rate+=0.2
                        #优先攻击后排小兵
                        if hit_target_info is not None and hit_target_info == furthest_enemy_soldier_runtime_id:
                            balance_rate+=0.1
                
                    if self.m_last_frame_soldier_av_hp == 0:
                             balance_rate=0
                    for soldier in self.enemy_soldiers:
                        total_hp+=soldier['hp']/soldier['max_hp']
                    
                    average_hp = self.m_last_frame_soldier_av_hp-total_hp/len(self.enemy_soldiers)

                    reward_struct.cur_frame_value = average_hp*balance_rate
                    #print(reward_struct.cur_frame_value)
    
    
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
    def calculate_forward(self, main_hero, main_tower, enemy_tower,main_spring):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        main_spring_pos = (main_spring["location"]["x"], main_spring["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        dist_hero2spring = math.dist(hero_pos, main_spring_pos)
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        dist_main2spring = math.dist(main_spring_pos,main_tower_pos)
        #战场前
        if dist_hero2emy > dist_main2emy:
            #进入战场
            if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99:
                forward_value += (dist_main2emy - dist_hero2emy) / dist_main2emy*100
                #print("进入战场")
            #回泉水
            if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] <0.3: 
                forward_value += -dist_hero2spring/dist_main2spring *0.01
                #print("回泉水")
        #战场中
        else:
            #大幅度减血鼓励撤退
            check_hp = self.check_hp(self.last_few_frame_hp)
            if check_hp<0 or enemy_tower['attack_target'] == self.main_hero_player_id :
                forward_value += (dist_hero2emy - dist_main2emy)/ dist_main2emy
                #print("大幅度减血鼓励撤退")
             #撤离战场
            if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] <0.3:
                forward_value +=(dist_hero2emy - dist_main2emy)/dist_main2emy*math.exp(-main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"])
                #print("撤离战场")
            if self.main_soldiers != []:
                s_pos = (self.main_soldiers[0]["location"]["x"],self.main_soldiers[0]["location"]["z"])
                dist_s2emy = math.dist(s_pos,enemy_tower_pos)
                if dist_hero2emy >8800 and dist_hero2emy >=dist_s2emy:
                    #鼓励朝塔前进
                    forward_value += (dist_hero2emy-dist_s2emy)/dist_main2emy*500
                    #print("鼓励朝塔前进")
       
            #鼓励守塔
        if self.enemy_soldiers!=[] and len(self.enemy_soldiers)>=2:
            es_pos = (self.enemy_soldiers[0]["location"]["x"],self.enemy_soldiers[0]["location"]["z"])
            dist_es2main = math.dist(es_pos,main_tower_pos)
            if dist_es2main<9000 and dist_hero2emy < dist_main2emy-9000:
                forward_value +=(dist_hero2emy - dist_main2emy ) / dist_main2emy
                #print("鼓励守塔")
       
        return forward_value
    
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
                #self.last_few_frame_hp.append(hero["actor_state"]["hp"])
                #print(hero["actor_state"]["hp"]/hero["actor_state"]["max_hp"])
                if hero["actor_state"]["hp"]== 0:
                    self.last_few_frame_hp = deque(maxlen=8)
                else:
                    self.last_few_frame_hp.append(hero["actor_state"]["hp"]/hero["actor_state"]["max_hp"])
                #print(self.last_few_frame_hp)
                if self.enemy_soldiers != []:
                    total_hp = 0
                    for soldier in self.enemy_soldiers:
                        total_hp+=soldier['hp']/soldier['max_hp']
                    self.m_last_frame_soldier_av_hp = total_hp/len(self.enemy_soldiers)
                else:self.m_last_frame_soldier_av_hp = 0
                #print(self.m_last_frame_soldier_av_hp)
            else:
                enemy_camp = hero["actor_state"]["camp"]
#######################################################################

#################################################################
    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def calculate_distance(self,loc1, loc2):
        return math.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['z'] - loc2['z'])**2)

   
    
    # 判断是否连续多帧大幅度掉血
    def check_hp(self, hp_queue):
        if len(hp_queue) >1 and hp_queue[len(hp_queue)-1] <= 0.4 * hp_queue[0]:
            return hp_queue[len(hp_queue)-1]-hp_queue[0]
        elif len(hp_queue) >1 and hp_queue[len(hp_queue)-2] >= 1.2 * hp_queue[0]:
            return hp_queue[len(hp_queue)-2] - hp_queue[0]
        else:
            return 0

    
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
                #(f'{reward_name}:{reward_struct.value}\n')
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
                    reward_struct.value /=50
            
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value/1000
                #print(f'{reward_name}:{reward_struct.value}\n')
            
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
                

            elif reward_name =='heal':
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if_use = reward_struct.cur_frame_value - reward_struct.last_frame_value
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                
                if  int(if_use):
                    if main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]>0.85:
                        balance_rate -= 0.3+main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]
                
                #print(main_hero['buff_state']['buff_marks'])
                else:
                    balance_rate = 0
                check_hp = self.check_hp(self.last_few_frame_hp)
                #print(check_hp)
                if check_hp > 0:
                    balance_rate += self.check_hp(self.last_few_frame_hp)
                reward_struct.value = balance_rate
                
                #print(f'{reward_name}:{reward_struct.value}')

            elif reward_name == "skill_hit_count":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value



            elif reward_name == "HurtToHero":

                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*100
                #print(f'{reward_name}:{reward_struct.value}\n')
                

            elif reward_name == "BeHurtByHero":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                #distance_to_enemy = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*50
                #print(f'{reward_name}:{reward_struct.value}')
            
            elif reward_name == "HurtToOthers":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*50
                #print(f'{reward_name}:{reward_struct.value}')
            
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value =(reward_struct.cur_frame_value - reward_struct.last_frame_value) 
                #print(f'{reward_name}:{reward_struct.value}')

            
            elif reward_name == "enemy_Soldiers_hp":
                
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate*100
                #print(f'{reward_name}:{reward_struct.value}\n')
            
            elif reward_name == "money":
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)/100
            
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
        # print('#################Current Reward##############')
        # for reward_name in reward_dict.keys():
        #     print(f'{reward_name}:{reward_dict[reward_name]}\n')

        reward_dict["reward_sum"] = reward_sum
        #print(reward_sum)





