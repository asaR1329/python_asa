#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('nbagg') #アニメーション表示の宣言
import matplotlib.animation as anm #アニメーションを扱うためのモジュール
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np


# In[2]:


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval
        
    def append(self,obj):
        self.objects.append(obj)
        
    def draw(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)
        
        elems = []
        
        if self.debug:
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:            
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
        plt.show()

    def one_step(self, i, elems, ax): #アニメーションを1コマ進めるメソッド
        while elems: elems.pop().remove() #elemsのリストをクリア
        time_str = "t = %.2f[s]" % (self.time_interval*i) #時刻として表示する文字列
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


# In[3]:


class IdealRobot:
    def __init__(self, pose, agent = None, sensor = None, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose] #軌跡の描画
        self.sensor = sensor
        
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2]) #センサ値を得た時刻の姿勢を渡す
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax,elems)
    
    @classmethod
    def state_transition(self, nu, omega, time, pose): #移動後の姿勢を返す
        t0 = pose[2]
        if math.fabs(omega) < 1e-10: #各速度で場合分け
            return pose + np.array( [nu*math.cos(t0),
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )
        
    def one_step(self, time_interval): #decisionメソッドを動かしてロボットを動かすメソッド
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


# In[4]:


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        
    def decision(self, observation=None):
        return self.nu, self.omega


# In[5]:


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None
    
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# In[6]:


class Map:
    def __init__(self):
        self.landmarks = [] #空のランドマークリストを準備
        
    def append_landmark(self, landmark): #ランドマークを追加
        landmark.id = len(self.landmarks) #追加するランドマークにIDを与える
        self.landmarks.append(landmark)
    
    def draw(self, ax, elems): #描画(Landmarkのdrawを順に呼び出し)
        for lm in self.landmarks: lm.draw(ax, elems)


# In[7]:


class IdealCamera:
    def __init__(self, env_map,                  distance_range=(0.5, 6.0),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []
        
        self.distance_range = distance_range
        self.direction_range = direction_range
    
    def visible(self, polarpos): #ランドマークが計測できる条件
        if polarpos is None:
            return False
        
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1]                 and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]
        
    def data(self, cam_pose): #すべてのランドマークの観測結果を返すためのメソッド
        observed = [] #センサ値を入れるリスト
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z,lm,id))
            
        self.lastdata = observed
        return observed
    
    @classmethod
    def observation_function(cls, cam_pose, obj_pos): #cam_pose:カメラの姿勢，obj_pos:物体の位置
        diff = obj_pos - cam_pose[0:2] #mj,x-x, mj,y-yを要素に持つベクトル
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2] #φの計算
        while phi >= np.pi: phi -= 2*np.pi #正規化
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi] ).T #ljの計算    
    
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")


# In[8]:


if __name__ == '__main__':
    world = World(30, 1)

    #地図を生成してランドマークを追加
    m = Map()
    m.append_landmark(Landmark( 2,-2))
    m.append_landmark(Landmark(-1,-3))
    m.append_landmark(Landmark( 3, 3))
    world.append(m)

    #ロボットを作成
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*math.pi)
    robot1 = IdealRobot( np.array([ 2, 3, math.pi/6]).T, sensor=IdealCamera(m), agent=straight) #ロボットのインスタンス生成
    robot2 = IdealRobot( np.array([-2,-1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
    world.append(robot1) #ロボットを生成
    world.append(robot2)

    world.draw()


# In[9]:


#原点から0.1m/sで1s直進
IdealRobot.state_transition(0.1, 0.0, 1.0, np.array([0, 0, 0]).T)


# In[10]:


#原点から0.1m/s,10deg/sで9s移動
IdealRobot.state_transition(0.1, 10.0/180*math.pi, 9.0, np.array([0, 0, 0]).T)


# In[11]:


#原点から0.1m/s，10deg/sで18s移動
IdealRobot.state_transition(0.1, 10.0/180*math.pi, 18.0, np.array([0, 0, 0]).T)

