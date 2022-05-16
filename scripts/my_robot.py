#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../scripts/')
from my_ideal_robot import*
from scipy.stats import expon, norm, uniform


# In[2]:


class Robot(IdealRobot):
    
    def __init__(self,pose,agent=None,sensor=None,color="black",
                 noise_per_meter=5,noise_std=math.pi/60, #小石の数=5/m, 踏んだ時発生するノイズ=3deg
                 bias_rate_stds=(0.1,0.1), #バイアスの係数をドローするための正規分布の標準偏差,それぞれ標準偏差10%で呼び出す
                 expected_stuck_time=1e100,expected_escape_time=1e-100, #スタックまでの時間の期待値，スタックから脱出するまでの時間の期待値
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)): #誘拐までの時間の期待値，誘拐後に置かれるロボットの位置の範囲
        super().__init__(pose,agent,sensor,color) #IdealRobotのinitを呼び出す
        self.noise_pdf=expon(scale=1.0/(1e-100+noise_per_meter)) #指数分布のオブジェクトを生成, scale=小石を踏むまでの平均の道のり
        self.distance_until_noise=self.noise_pdf.rvs() #最初に小石を踏むまでの道のりをセット, rvsはドローのためのメソッド
        self.theta_noise=norm(scale=noise_std) #θに加える雑音を決めるための正規分布のオブジェクトを作成
        self.bias_rate_nu=norm.rvs(loc=1.0,scale=bias_rate_stds[0]) #ロボット固有のバイアスを決定
        self.bias_rate_omega=norm.rvs(loc=1.0,scale=bias_rate_stds[1]) #同上
        
        self.stuck_pdf=expon(scale=expected_stuck_time) #スタックの確率密度関数
        self.escape_pdf=expon(scale=expected_escape_time) #脱出の確率密度関数
        self.time_until_stuck=self.stuck_pdf.rvs() #時間の初期化
        self.time_until_escape=self.escape_pdf.rvs() #同上
        self.is_stuck=False #ロボットがスタック中かどうか
        
        self.kidnap_pdf = expon(scale=expected_kidnap_time) #指数分布
        self.time_until_kidnap = self.kidnap_pdf.rvs() #誘拐までの時間を管理する関数
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0],ry[0],0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi)) #ロボットの姿勢をドローするための一様分布
        
    def noise(self,pose,nu,omega,time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*omega*time_interval #次に小石を踏むまでの道のりを経過時間の分だけ減らす
        if self.distance_until_noise <= 0.0: #小石を踏んだかどうか判定
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs() #θに雑音を加える
        
        return pose
    
    def bias(self, nu, omega): 
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega
    
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:            
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck)
    
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose
            
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, omega, nu, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)


# In[3]:


class Camera(IdealCamera):
    def __init__(self, env_map,
                distance_range = (0.5, 6.0),
                direction_range = (-math.pi/3, math.pi/3),
                distance_noise_rate = 0.1, direction_noise = math.pi/90,
                distance_bias_rate_stddev=0.1, direction_bias_stddev=math.pi/90, #バイアスを決定するときの正規分布の標準偏差
                phantom_prob=0.0, phantom_range_x=(-5.0,5.0), phantom_range_y=(-5.0,5.0),
                oversight_prob=0.1, occlusion_prob=0.0):
        super().__init__(env_map, distance_range, direction_range)
        
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale=direction_bias_stddev)
        
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1]-rx[0], ry[1]-ry[0]))
        self.phantom_prob = phantom_prob
        
        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob
        
    def noise(self, relpos):   
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T
    
    def bias(self, relpos): #バイアスをセンサ値に加えるメソッド
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std,
                                 self.direction_bias]).T
    
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_funtion(cam_pose, pos)
        else:
            return relpos
        
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos
    
    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs()*(self.distance_range[1] - relpos[0])
            phi = relpos[1]
            return np.array([ell, relpos[1]]).T
    
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)    
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))
            
        self.lastdata = observed 
        return observed


# In[4]:


if __name__ == '__main__': 
    world = World(30, 0.1, debug=False)  

    m = Map()                                  
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)          

    straight = Agent(0.2, 0.0)    
    circling = Agent(0.2, 10.0/180*math.pi)  
    r = Robot( np.array([ 2, 2, math.pi/6]).T, sensor=Camera(m, occlusion_prob=0.1), agent=circling) 
    world.append(r)

    
    world.draw()


# In[ ]:




