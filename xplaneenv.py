import time
from XPlaneResearch.AircraftPlant import AircraftPlant, AttitudeController
from XPlaneUDP.XPlaneUdp import XPlaneUdp
from pynput.mouse import Button, Controller
mouse = Controller()
import numpy as np

class XPlaneEnv():
    def __init__(self):
        print("Staring LazyHawk on simulator")
        plant = AircraftPlant()
        print("Will reload situation")
        # plant.reset_flight()

        print("Wait for simuation loading")
        self.att_controller = AttitudeController(debug=True)
        plant.control_callback = lambda plant, dt: self.att_controller.control(plant, dt)
        self.plant = plant
        self.reset()
        self.total_eneygy_last = None

    def state(self):
        state =  [self.plant.airspeed, self.plant.ver_vel_ind,
                  self.plant.euler[2], self.plant.euler[1],
                  self.plant.ax_body, self.plant.ay_body, self.plant.az_body, self.plant.data["yoke_pitch"]]
        return state

    def update(self):
        self.plant.update()

    def step(self, action):
        print(action)
        roll_sp = (action - 3)/3 * 30 / 57.3
        print(f"DQN ROLLSP {roll_sp*57.3:3.1f}")
        self.att_controller.set_attitude_sp(roll_sp, 0)
        # time.sleep(0.01)

        self.plant.update()
        return self.state(), self.reward(), self.done(), 0

    def reward(self):
        return self.plant.ver_vel_ind

    def reset(self):
        mouse.click(Button.left, 2)
        print("Reseting...")
        self.plant.load_situation()
        self.att_controller.reset()
        time.sleep(1)
        self.plant.set_ctrl(0, 0 ,0)
        time.sleep(3)
        return self.state()



    def done(self):
        return False

class XPlaneEnvCon():
    def __init__(self):
        print("Staring LazyHawk on simulator")
        plant = AircraftPlant()
        print("Will reload situation")
        # plant.reset_flight()

        print("Wait for simuation loading")
        self.att_controller = AttitudeController(debug=True)
        plant.control_callback = lambda plant, dt: self.att_controller.control(plant, dt)
        self.plant = plant
        self.ls_time = None
        # self.reset()

    def state(self):
        state =  [self.plant.airspeed, self.plant.ver_vel_ind,
                  self.plant.euler[2], self.plant.euler[1],
                  self.plant.angular_rate[0], self.plant.angular_rate[1], self.plant.angular_rate[2],
                  self.plant.ax_body, self.plant.ay_body, self.plant.az_body, self.plant.data["yoke_pitch"]]
        return state

    def update(self):
        self.plant.update()

    def step(self, action):
        # print("Step")
        # print(action)
        roll_sp = np.clip(action[0], -1, 1) * 45 / 57.3
        pitch_sp = np.clip(action[1], -1, 1) * 30 / 57.3

        #Donothing Test
        # roll_sp = 0
        # pitch_sp = 0
        # print(f"DQN ROLLSP {roll_sp*57.3:3.1f} PITCHSP {pitch_sp*57.3:3.1f}")


        ts = time.time()
        self.plant.update()
        # print(f"Update use time {(time.time() - ts)*1000:4.1f}ms")
        self.att_controller.set_attitude_sp(roll_sp, pitch_sp)
        # time.sleep(0.01)

        if self.ls_time is None:
            self.ls_time = time.time()
        else:
            tnow = time.time()
            # print(f"L2N {(tnow - self.ls_time)*1000:3.1f}ms" )
            # if tnow - self.ls_time < 0.02:
            #     time.sleep(self.ls_time - tnow + 0.02)
            self.ls_time = tnow
        return self.state(), self.reward(), self.done(), 0

    def total_energy(self):
        print(f"AIRSPD {self.plant.airspeed}")
        return self.plant.alt * 9.8 + 0.5*self.plant.airspeed*self.plant.airspeed

    def reward_te(self):
        if self.total_energy_last is None:
            self.total_energy_last = self.total_energy()
            self.total_energy_last_time = time.time()
            return 0
        else:
            te = self.total_energy()
            te_last = self.total_energy_last
            t_last = self.total_energy_last_time
            self.total_energy_last = te
            self.total_energy_last_time = time.time()
            dt  = self.total_energy_last_time - t_last

            self.total_energy_last_time = t_last
            return (te - te_last) / dt

    def reward(self):
        # print(self.plant.ver_vel_ind)
        # return self.plant.ver_vel_ind
        return self.plant.ver_vel_ind

    def reset(self):
        mouse.click(Button.left, 2)

        # self.plant.pause()
        self.plant.set_ctrl(0, 0, 0)

        print("Reseting....")
        self.plant.load_situation()
        time.sleep(2)
        self.plant.set_ctrl(0, 0, 0)

        # self.plant.pause()
        self.att_controller.reset()
        self.att_controller.set_attitude_sp(0, 0)
        for i in range(100):
            self.step([0, 0])
        return self.state()

    def altitude(self):
        return self.plant.alt



    def done(self):
        return False
