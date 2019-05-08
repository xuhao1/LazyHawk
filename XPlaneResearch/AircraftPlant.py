from XPlaneUDP.XPlaneUdp import XPlaneUdp
import numpy as np
import math
import time
from pymavlink import quaternion
import keyboard

def opengl2NED(vec):
    vec = vec.copy()
    return [-vec[2], vec[0], vec[1]]


class AircraftPlant():
    def __init__(self):
        print("Opening XPLANE UDP")
        xp = self.xp = XPlaneUdp()
        xp.defaultFreq = 100
        beacon = xp.FindIp()
        assert beacon
        self.data = dict()
        self.dataref_to_data = dict()

        print("XPLANE UDP Start successul, adding datarefs")

        self.AddDataRef("sim/flightmodel/position/indicated_airspeed", "indicated_airspeed")
        self.AddDataRef("sim/flightmodel/position/true_airspeed", "airspeed")
        self.AddDataRef("sim/flightmodel/position/Prad", "p")
        self.AddDataRef("sim/flightmodel/position/Qrad", "q")
        self.AddDataRef("sim/flightmodel/position/Rrad", "r")
        self.AddDataRef("sim/flightmodel/position/alpha", "aoa")
        self.AddDataRef("sim/flightmodel/position/beta", "sideslip")
        self.AddDataRef("sim/flightmodel/position/true_theta", "theta")
        self.AddDataRef("sim/flightmodel/position/true_phi", "phi")
        self.AddDataRef("sim/flightmodel/position/true_psi", "psi")
        self.AddDataRef("sim/flightmodel/position/q[0]", "quat0")
        self.AddDataRef("sim/flightmodel/position/q[1]", "quat1")
        self.AddDataRef("sim/flightmodel/position/q[2]", "quat2")
        self.AddDataRef("sim/flightmodel/position/q[3]", "quat3")
        self.AddDataRef("sim/flightmodel2/wing/aileron1_deg[0]", "aileron")
        self.AddDataRef("sim/flightmodel2/wing/elevator1_deg[0]", "elevator")
        self.AddDataRef("sim/flightmodel2/wing/rudder1_deg[0]", "rudder")
        self.AddDataRef("sim/joystick/yoke_pitch_ratio", "yoke_pitch")
        self.AddDataRef("sim/joystick/yoke_roll_ratio", "yoke_roll")
        self.AddDataRef("sim/joystick/yoke_heading_ratio", "yoke_yaw")
        self.AddDataRef("sim/time/total_running_time_sec", "running_time")
        self.AddDataRef("sim/flightmodel/position/vh_ind", "VVI")
        self.AddDataRef("sim/flightmodel/misc/h_ind", "alt")

        # Coor is different as we general use
        # # it seem source VX , VY ,VZ is (general) Vy Vz Vx

        self.AddDataRef("sim/flightmodel/position/local_ax", "local_ax")
        self.AddDataRef("sim/flightmodel/position/local_ay", "local_ay")
        self.AddDataRef("sim/flightmodel/position/local_az", "local_az")

        self.AddDataRef("sim/flightmodel/position/local_vx", "local_vx")
        self.AddDataRef("sim/flightmodel/position/local_vy", "local_vy")
        self.AddDataRef("sim/flightmodel/position/local_vz", "local_vz")

        self.aoa = 0
        self.sideslip = 0
        self.airspeed = 0
        self.ind_airspeed = 0
        self.euler = np.array([0, 0, 0])
        self.body_vel = np.array([0, 0, 0])
        self.angular_rate = np.array([0, 0, 0])
        self.quat = quaternion.Quaternion([1, 0, 0, 0])
        self.aileron_deg = 0
        self.elevator_deg = 0
        self.rudder_deg = 0
        self.control_callback = None
        self.running_time = 0
        self.dt = 0
        self.ver_vel_ind = 0
        self.alt = 0
        self.data_record = []
        self.vx_body = 0
        self.vy_body = 0
        self.vz_body = 0

        self.ax_body = 0
        self.ay_body = 0
        self.az_body = 0
        # self.vjoy = pyvjoy.VJoyDevice(1)

    def AddDataRef(self, datarefname, keyname, default=0):
        xp = self.xp
        xp.AddDataRef(datarefname, freq=100)
        self.dataref_to_data[datarefname] = keyname
        self.data[keyname] = default

    def recv(self):
        values = self.xp.GetValues()
        for k in values:
            v = values[k]
            self.data[self.dataref_to_data[k]] = v

    def update(self):
        self.recv()
        self.update_state()
        if self.control_callback is not None:
            self.control_callback(self, self.dt)

    def global2bodyNED(self, vec):
        vec = opengl2NED(vec)
        vec = self.quat.inversed.transform(vec.copy())
        vec[1] = - vec[1]
        vec[2] = vec[2]
        return vec

    def update_state(self):
        self.data_record.append(self.data.copy())
        self.aoa = self.data['aoa']
        self.sideslip = self.data['sideslip']
        self.airspeed = self.data['airspeed']
        self.ind_airspeed = self.data['indicated_airspeed'] * 0.514444  # Source data is knots
        # Euler Yaw Pitch Roll
        self.euler = np.array([self.data['psi'], self.data['theta'], self.data['phi']]) * math.pi / 180
        self.angular_rate = np.array([self.data['p'], self.data['q'], self.data['r']])
        self.quat = quaternion.Quaternion(
            [self.data['quat0'], self.data['quat1'], self.data['quat2'], self.data['quat3']])
        self.aileron_deg = self.data['aileron']
        self.elevator_deg = self.data['elevator']
        self.rudder_deg = self.data['rudder']
        self.alt = self.data["alt"] * 0.3048
        if self.running_time != 0:
            self.dt = self.data['running_time'] - self.running_time
        else:
            self.dt = 0.02
        self.running_time = self.data['running_time']
        self.ver_vel_ind = self.data['VVI']

        new_acc = self.global2bodyNED([self.data["local_ax"], self.data["local_ay"] + 9.8, self.data["local_az"]])
        self.ax_body = new_acc[0]
        self.ay_body = new_acc[1]
        self.az_body = new_acc[2]
        body_v = self.global2bodyNED([self.data["local_vx"], self.data["local_vy"], self.data["local_vz"]])
        # print("BODY vec {:3.1f} {:3.1f} {:3.1f} , Acc {:3.1f} {:3.1f} {:3.1f}".format(
        #     body_v[0], body_v[1], body_v[2],
        #     new_acc[0], new_acc[1], new_acc[2]))

        self.data["vx_body"] = body_v[0]
        self.data["vy_body"] = body_v[1]
        self.data["vz_body"] = body_v[2]
        self.vx_body, self.vy_body, self.vz_body = body_v[0], body_v[1], body_v[2]

        self.data["ax_body"] = self.ax_body
        self.data["ay_body"] = self.ay_body
        self.data["az_body"] = self.az_body

        # self.data["local_vx"], self.data["local_vy"], self.data["local_vz"] = local_v[0], local_v[1], local_v[2]
    def set_ail_ctrl(self, ail):
        self.xp.WriteDataRef("sim/joystick/yoke_roll_ratio", value=ail)

    def set_ctrl(self, ail, ele, rud):
        # DREF_YokePitch
        self.override_ctrl(True)
        self.xp.WriteDataRef("sim/joystick/yoke_roll_ratio", value=ail)
        self.xp.WriteDataRef("sim/joystick/yoke_pitch_ratio", value=ele)
        self.xp.WriteDataRef("sim/joystick/yoke_heading_ratio", value=rud)
        pass

    def pause(self):
        keyboard.press_and_release("p")

    def load_situation(self):
        # self.xp.load_situation()
        # keyboard.press_and_release("ctrl+shift+f11")
        keyboard.press_and_release("ctrl+shift+f12")

    def reset_flight(self):
        self.xp.reset_flight()

    def override_ctrl(self, override):
        self.xp.WriteDataRef("sim/operation/override/override_joystick", override, vtype='bool')
        pass

    def trivial_level_ctrl(self, dt, trim_pitch_deg=0):
        self.override_ctrl(True)
        kp_roll = 1.0
        kp_pitch = 2.0
        trim_pitch_deg = trim_pitch_deg * math.pi / 180
        self.set_ctrl(-kp_roll * self.euler[2], kp_pitch * (trim_pitch_deg - self.euler[1]), 0)

    def save_data_to_cifer_format(self, data_name_list, cifer_name_list, start_time=0, end_time=0):
        arr = self.save_data_to_arr(data_name_list, start_time, end_time)
        data_len = len(arr[:, 0])
        print(arr)
        print("data time {} len {} save to cifer".format(end_time - start_time, data_len))
        sampled_delta = (end_time - start_time) / data_len
        cifer_file = "{0} {1}\n".format(sampled_delta, data_len)
        for name in cifer_name_list:
            cifer_file = cifer_file + "{}\t".format(name)
        cifer_file = cifer_file + "\n"
        for i in range(data_len):
            for j in range(data_name_list.__len__()):
                cifer_file = cifer_file + "{}\t".format(arr[i][j])
            cifer_file = cifer_file + "\n"
        # print(cifer_file)
        return cifer_file

    def save_data_to_arr(self, data_name_list, start_time=0, end_time=0):
        # Locate start and end time
        start_ptr = 0
        end_ptr = len(self.data_record)

        assert end_ptr > 1, "no data record! Can't output"
        assert data_name_list.__len__() > 1, "No data name list!"
        print("Try to save data from {:4.1f} to {:4.1f} whole record{}".format(start_time, end_time,
                                                                               len(self.data_record)))
        if start_time > 0:
            # Locate start ptr
            for i in range(self.data_record.__len__()):
                data_set = self.data_record[i]
                if data_set["running_time"] > start_time:
                    start_ptr = i
                    # Found start ptr, just break
                    print(
                        "Req {0} Found ptr {1}, time {2:4.2f}".format(start_time, start_ptr, data_set["running_time"]))
                    break
        if end_time > 0:
            for j in range(start_ptr, self.data_record.__len__()):
                data_set = self.data_record[j]
                if data_set["running_time"] > end_time:
                    end_ptr = j
                    # Found end ptr, just break
                    break
            print("Req {:4.2f} Found ptr {}, time {:4.2f}".format(end_time, end_ptr,
                                                                  (self.data_record[end_ptr - 1])["running_time"]))
        # Now convert data to arr
        arr_output = []
        for i in range(start_ptr, end_ptr):
            col = []
            data_set = self.data_record[i]
            for name in data_name_list:
                col.append(data_set[name])
            arr_output.append(col)
        return np.array(arr_output)


def float_const(v, low, up):
    if low > v:
        return low
    if up < v:
        return up
    return v


import matplotlib.pyplot as plt

class AttitudeController:
    def __init__(self, debug=False):
        self.roll_err_int = 0
        # self.kp_roll = 15
        # self.kp_roll_rate = 3
        # self.ki_roll = 1

        self.kp_roll = 5
        self.kp_roll_rate = 1
        self.ki_roll = 0.5

        self.pitch_err_int = 0
        # self.kp_pitch = 15.0
        # self.kpitch_rate = 5
        # self.ki_pitch = 10
        self.kp_pitch = 5.0
        self.kpitch_rate = 1.5
        self.ki_pitch = 1.0
        # self.ki_pitch = 5.0

        self.err_pitch = 0

        self.alt_err_int = 0
        self.alt_sp = 0
        self.alt_kp = 0.1
        self.alt_kd = 0.15

        self.time = 0
        self.tick = 0

        self.sweep_start_time = 0

        self.sweep_t_series = []
        self.sweep_ele_series = []

        self.sweep_speed_series = []
        self.sweep_vert_ver_series = []
        self.sweep_q_series = []
        self.sweep_theta_series = []
        self.sweep_aoa_series = []
        self.sweep_alt_series = []
        self.sweep_phi_series = []
        self.sweep_vx_series = []
        self.sweep_vy_series = []
        self.sweep_vz_series = []

        self.debug = debug

        self.forward_ail = 0
        self.forward_ele = 0
        self.forward_rud = 0

        self.roll_sp = 0
        self.pitch_sp = 0

    def set_forward(self, ail, ele, rud=0):
        self.forward_ail = ail
        self.forward_ele = ele
        self.forward_rud = rud

    def set_attitude_sp(self, roll_sp, pitch_sp):
        # print(roll_sp, pitch_sp)

        self.roll_sp = roll_sp * 0.1 + self.roll_sp*0.9
        self.pitch_sp = pitch_sp * 0.1 + self.pitch_sp * 0.9
        # self.roll_sp = 0
        # self.pitch_sp = 0
        # print(f"RSP {self.roll_sp*57.3:4.1f} {self.pitch_sp*57.3:4.1f}")

    def record_inf(self, ele):
        self.sweep_t_series.append(self.time)
        self.sweep_ele_series.append(ele)
        self.sweep_speed_series.append(plant.airspeed)
        self.sweep_aoa_series.append(plant.aoa)
        self.sweep_theta_series.append(math.degrees(plant.euler[1]))
        self.sweep_vert_ver_series.append(plant.ver_vel_ind)
        self.sweep_alt_series.append(plant.alt)
        self.sweep_q_series.append(math.degrees(plant.angular_rate[1]))
        self.sweep_phi_series.append(math.degrees(plant.euler[2]))
        self.sweep_vx_series.append(plant.ax_body)
        self.sweep_vy_series.append(plant.az_body)
        self.sweep_vz_series.append(plant.vz_body)

        pass

    def plot_record(self):

        plt.ion()
        plt.figure(255)
        plt.clf()
        plt.subplot(421)
        plt.plot(self.sweep_t_series, self.sweep_ele_series)
        plt.title('Elevator')
        plt.grid(True)

        plt.subplot(422)
        plt.plot(self.sweep_t_series, self.sweep_q_series)
        plt.title('PitchAngularRate deg/s')
        plt.grid(True)

        plt.subplot(423)
        plt.plot(self.sweep_t_series, self.sweep_theta_series)
        plt.title('PitchAngle deg')
        plt.grid(True)

        # plt.subplot(424)
        # plt.plot(self.sweep_t_series, self.sweep_aoa_series)
        # plt.title('PitchAoa deg')
        # plt.grid(True)
        #
        # plt.subplot(425)
        # plt.plot(self.sweep_t_series, self.sweep_vert_ver_series)
        # plt.title('VVI m/s')
        # plt.grid(True)

        plt.subplot(424)
        plt.plot(self.sweep_t_series, self.sweep_vy_series)
        plt.title('az')
        plt.grid(True)

        plt.subplot(425)
        plt.plot(self.sweep_t_series, self.sweep_vz_series)
        plt.title('vz')
        plt.grid(True)
        #
        # plt.subplot(425)
        # plt.plot(self.sweep_t_series, self.sweep_vz_series)
        # plt.title('Vz')
        # plt.grid(True)

        plt.subplot(426)
        plt.plot(self.sweep_t_series, self.sweep_alt_series)
        plt.title('alt m')
        plt.grid(True)

        plt.subplot(427)
        plt.plot(self.sweep_t_series, self.sweep_phi_series)
        plt.title('Roll angle')
        plt.grid(True)

        plt.subplot(428)
        plt.plot(self.sweep_t_series, self.sweep_speed_series)
        plt.title('TrueAirspeed m/s')
        plt.grid(True)

        plt.show()

    def roll_ctrl(self, plant: AircraftPlant, dt, roll_sp):
        kp_roll = self.kp_roll
        ki_roll = self.ki_roll
        kp_roll_rate = self.kp_roll_rate
        err = roll_sp - plant.euler[2]
        self.roll_err_int = self.roll_err_int + err * dt
        self.roll_err_int = float_const(self.roll_err_int, -0.3 / ki_roll, 0.3 / ki_roll)
        ail = kp_roll * err + kp_roll_rate * (-plant.angular_rate[0]) + ki_roll * self.roll_err_int
        return ail

    def pitch_control(self, plant: AircraftPlant, dt, pitch_sp=0):
        kp = self.kp_pitch
        ki = self.ki_pitch
        err = pitch_sp - plant.euler[1]

        self.err_pitch = err
        self.pitch_err_int = float_const(self.pitch_err_int + dt * err, -0.2 / ki, 0.2 / ki)
        ele = kp * err + ki * self.pitch_err_int + self.kpitch_rate * -plant.angular_rate[1]
        # print(ele)

        return float_const(ele, -1, 1)

    def control(self, plant: AircraftPlant, dt):
        # print("dt{0}".format(dt))
        self.tick = self.tick + 1
        self.time = plant.running_time
        ail = self.roll_ctrl(plant, dt, self.roll_sp)
        ele = self.pitch_control(plant, dt, self.pitch_sp)
        rud = 0

        # print(f"SP R{self.roll_sp*57.3:3.1f} P{self.pitch_sp*57.3:3.1f};R {plant.euler[2]*57.3:3.1f} P {plant.euler[1]*57.3:3.1f}")
        plant.set_ctrl(ail + self.forward_ail, ele + self.forward_ele, rud + self.forward_rud)

    def reset(self):
        self.roll_err_int = 0
        self.pitch_err_int = 0
        pass

if __name__ == "__main__":
    plant = AircraftPlant()
    plant.control_callback = AircraftPlant.trivial_level_ctrl
    start = time.time()
    tick = 0
    while True:
        tick = tick + 1
        plant.update()
        # plant.trivial_level_ctrl(15)
        sec = (time.time() - start)
        print("averagetime : {:3.2f}ms dt {:3.2f}ms".format(sec / tick * 1000, plant.dt * 1000))
        # print("time {0:2f} ail {1:2f} deg {2:2f} ele{3:2f} {4:3f}".format(sec,plant.data["yoke_roll"], plant.aileron_deg,plant.data["yoke_pitch"],plant.airspeed))
