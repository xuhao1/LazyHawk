import time
from XPlaneResearch.AircraftPlant import AircraftPlant, AttitudeController
from XPlaneUDP.XPlaneUdp import XPlaneUdp



def reward_func(plant):
    return plant.ver_vel_ind

if __name__ == "__main__":

    print("Staring LazyHawk on simulator")
    plant = AircraftPlant()
    print("Will reload situation")
    plant.load_situation()
    # plant.reset_flight()

    print("Wait for simuation loading")
    # time.sleep(5)
    att_controller = AttitudeController(debug=True)
    plant.control_callback = lambda plant, dt: att_controller.control(plant, dt)
    start = time.time()
    tick = 0
    while True:
        att_controller.set_attitude_sp(0.1, 0.1)
        tick = tick + 1
        plant.update()
        # plant.trivial_level_ctrl(15)
        sec = (time.time() - start)
        print("averagetime : {:3.2f}ms dt {:3.2f}ms".format(sec / tick * 1000, plant.dt * 1000))
        xp = XPlaneUdp()
