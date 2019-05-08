import matplotlib

matplotlib.use("Qt5Agg")

from AircraftPlant import AircraftPlant
import time
import pygame
from generate_sweep_signal import sweep_signal_generater
import matplotlib.pyplot as plt
import math
import numpy as np
import datetime


def float_const(v, low, up):
    if low > v:
        return low
    if up < v:
        return up
    return v


class AltController:
    def __init__(self, debug=False):
        self.roll_err_int = 0
        self.kp_roll = 2.0
        self.kp_roll_rate = 0.8
        self.ki_roll = 0.1

        self.vertvel_err_int = 0
        self.kp_vert_vel = 0.04
        self.kd_vert_vel = 0.02
        self.kdd_vert_vel = 0
        self.ki_vert_vel = 0.005
        self.err_vertvel = 0

        self.alt_err_int = 0
        self.alt_sp = 0
        self.alt_kp = 0.1
        self.alt_kd = 0.15

        self.time = 0
        self.tick = 0
        self.enable_sweep_test = False
        self.inject_type = 0
        self.sweep_time = 290
        self.sweep_signal_time = 270
        self.sweep_prepare_time = 3
        # self.sweep_time = 5
        # self.sweep_signal_time = 2
        # self.sweep_prepare_time = 2
        # self.sweep_gen = sweep_signal_generater(self.sweep_signal_time,omgmin=0.02,omgmax=8)
        # self.sweep_gen = sweep_signal_generater(self.sweep_signal_time)
        self.sweep_gen = sweep_signal_generater(self.sweep_signal_time / 3, omgmin=0.1)

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

    def set_forward(self, ail, ele, rud=0):
        self.forward_ail = ail
        self.forward_ele = ele
        self.forward_rud = rud

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

    def roc_control(self, plant: AircraftPlant, dt, roc_sp=0):
        kp = self.kp_vert_vel
        ki = self.ki_vert_vel
        kd = self.kd_vert_vel
        err = roc_sp - plant.ver_vel_ind
        err_d = 0
        if dt > 1e-6:
            err_d = (err - self.err_vertvel) / dt

        self.err_vertvel = err
        self.vertvel_err_int = float_const(self.vertvel_err_int + dt * err, -0.3 / ki, 0.3 / ki)
        ele = kp * err + ki * self.vertvel_err_int + kd * err_d + self.kdd_vert_vel * -plant.angular_rate[1]

        return float_const(ele, -1, 1)

    def alt_control(self, plant: AircraftPlant, dt, alt_sp):
        err = alt_sp - plant.alt
        err_d = - plant.ver_vel_ind
        roc = err * self.alt_kp + err_d * self.alt_kd
        roc = float_const(roc, -10, 10)
        # print("alt sp {0:4.1f} now {3:4.1f} err{1:3.1f} rocsp {2:2.1f} now {4:2.1f}".format(alt_sp, err, roc, plant.alt,
        #                                                                                     plant.ver_vel_ind))
        return self.roc_control(plant, dt, roc_sp=roc)

    def control(self, plant: AircraftPlant, dt):
        # print("dt{0}".format(dt))
        self.tick = self.tick + 1
        self.time = plant.running_time
        ail = self.roll_ctrl(plant, dt, 0)

        if self.alt_sp == 0 and plant.alt > 0:
            self.alt_sp = plant.alt
            print("Will hold alt at {0:4.1f} meter {1:5.1f} fts".format(self.alt_sp, self.alt_sp / 0.3048))

        ele = 0
        if self.alt_sp > 0 and not self.enable_sweep_test:
            # when sweep, using roc control
            ele = self.alt_control(plant, dt, self.alt_sp)
        else:
            ele = self.roc_control(plant, dt, 0)

        rud = 0

        if self.enable_sweep_test:
            ail, ele, rud = self.inject_test_control(plant, ail, ele, rud)
            self.record_inf(ele)

        if self.debug:
            self.record_inf(ele)
            if self.tick % 100 == 0:
                self.plot_record()
        plant.set_ctrl(ail + self.forward_ail, ele + self.forward_ele, rud + self.forward_rud)

    def inject_test_control(self, plant: AircraftPlant, ail, ele, rud):
        t_sweep = self.time - self.sweep_start_time - self.sweep_prepare_time

        if 0 < t_sweep < self.sweep_signal_time and self.inject_type == 0:
            ele = ele + 0.1 * self.sweep_gen.get_value_by_time(t_sweep)
            ele = float_const(ele, -1, 1)
        else:
            self.ele_trim = ele
            pass
            # print("Finish Sweep,Waiting for return balance")
        if (self.time - self.sweep_start_time) > self.sweep_time:
            self.enable_sweep_test = False
            print("Finish Sweep,end inject")
            self.plot_record()
            save_data_list = ["running_time", "yoke_pitch",
                              "theta", "airspeed", "q", "aoa", "VVI", "alt",
                              "vx_body", "vy_body", "vz_body",
                              "ax_body", "ay_body", "az_body"]

            cifer_data_names = ["time", "yokeele", "theta", 'aspd', 'q', 'aoa', 'VVI', 'alt']
            arr = plant.save_data_to_arr(save_data_list, self.sweep_start_time, self.time)
            cifer_data = plant.save_data_to_cifer_format(save_data_list, cifer_data_names, self.sweep_start_time,
                                                         self.time)

            filename = "data/sweep_data_{0}".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
            np.save(filename, arr)

            f = open("{}.dat".format(filename), "w")
            f.write(cifer_data)

            print("save sweep time data to {0}.npy and {0}.dat".format(filename))

        return ail, ele, rud

    def toggle_sweep_test(self,inject_type = 0):
        self.enable_sweep_test = True
        self.inject_type = inject_type
        if self.inject_type == 1:
            self.sweep_time = 10
        # not self.enable_sweep_test
        if self.enable_sweep_test:
            print("Start Sweep test")
            self.sweep_start_time = self.time
        else:
            print("Disable sweep testp")


if __name__ == "__main__":
    pygame.init()
    pygame.joystick.init()

    t16000m = None
    twcs = None
    for k in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(k)
        print(joystick.get_name())
        if joystick.get_name() == "T.16000M":
            t16000m = joystick
            t16000m.init()
            print("Found Joystick {0}".format(t16000m.get_name()))
        if joystick.get_name() == "TWCS Throttle":
            twcs = pygame.joystick.Joystick(k)
            twcs.init()
            print("Found Joystick {0}".format(twcs.get_name()))

    plant = AircraftPlant()
    alt_controller = AltController(debug=True)
    plant.control_callback = lambda plant, dt: alt_controller.control(plant, dt)
    start = time.time()
    tick = 0
    while True:
        tick = tick + 1
        plant.update()
        event = pygame.event.get()
        if t16000m is not None and t16000m.get_button(13) > 0:
            alt_controller.toggle_sweep_test()
        if t16000m is not None and t16000m.get_button(14) > 0:
            alt_controller.toggle_sweep_test(inject_type=1)
        # alt_controller.set_forward(t16000m.get_axis(0), t16000m.get_axis(1))
        if alt_controller.inject_type == 1:
            alt_controller.set_forward(0, twcs.get_axis(3)*0.3)
