#!/usr/bin/env python

from .scservo_def import *
from .protocol_packet_handler import *
from .group_sync_read import *
from .group_sync_write import *

#波特率定义
SMS_STS_1M = 0
SMS_STS_0_5M = 1
SMS_STS_250K = 2
SMS_STS_128K = 3
SMS_STS_115200 = 4
SMS_STS_76800 = 5
SMS_STS_57600 = 6
SMS_STS_38400 = 7

#内存表定义
#-------EPROM(只读)--------
SMS_STS_MODEL_L = 3
SMS_STS_MODEL_H = 4

#-------EPROM(读写)--------
SMS_STS_ID = 5
SMS_STS_BAUD_RATE = 6
SMS_STS_RETURN_DELAY = 7
SMS_STS_RETURN_LEVEL = 8
SMS_STS_MIN_ANGLE_LIMIT_L = 9
SMS_STS_MIN_ANGLE_LIMIT_H = 10
SMS_STS_MAX_ANGLE_LIMIT_L = 11
SMS_STS_MAX_ANGLE_LIMIT_H = 12
SMS_STS_TEMP_LIMIT = 13
SMS_STS_MAX_VOLTAGE = 14
SMS_STS_MIN_VOLTAGE = 15
SMS_STS_MAX_TORQUE_L = 16
SMS_STS_MAX_TORQUE_H = 17
SMS_STS_UNLOAD_CONDITION = 19
SMS_STS_LED_ALARM = 20
SMS_STS_P_COEF = 21
SMS_STS_D_COEF = 22
SMS_STS_I_COEF = 23
SMS_STS_MIN_STARTUP_FORCE_L = 24
SMS_STS_MIN_STARTUP_FORCE_H = 25
SMS_STS_CW_DEAD = 26
SMS_STS_CCW_DEAD = 27
SMS_STS_PROTECT_CURRENT_L = 28
SMS_STS_PROTECT_CURRENT_H = 29
SMS_STS_ANGLE_RESOLUTION = 30
SMS_STS_OFS_L = 31
SMS_STS_OFS_H = 32
SMS_STS_MODE = 33
SMS_STS_PROTECT_TORQUE = 34
SMS_STS_PROTECT_TIME = 35
SMS_STS_OVERLOAD_TORQUE = 36
SMS_STS_OVERCURRENT_TIME = 38

#-------SRAM(读写)--------
SMS_STS_TORQUE_ENABLE = 40
SMS_STS_ACC = 41
SMS_STS_GOAL_POSITION_L = 42
SMS_STS_GOAL_POSITION_H = 43
SMS_STS_GOAL_TIME_L = 44
SMS_STS_GOAL_TIME_H = 45
SMS_STS_GOAL_SPEED_L = 46
SMS_STS_GOAL_SPEED_H = 47
SMS_STS_TORQUE_LIMIT_L = 48
SMS_STS_TORQUE_LIMIT_H = 49
SMS_STS_LOCK = 55

#-------SRAM(只读)--------
SMS_STS_PRESENT_POSITION_L = 56
SMS_STS_PRESENT_POSITION_H = 57
SMS_STS_PRESENT_SPEED_L = 58
SMS_STS_PRESENT_SPEED_H = 59
SMS_STS_PRESENT_LOAD_L = 60
SMS_STS_PRESENT_LOAD_H = 61
SMS_STS_PRESENT_VOLTAGE = 62
SMS_STS_PRESENT_TEMPERATURE = 63
SMS_STS_MOVING = 66
SMS_STS_PRESENT_CURRENT_L = 69
SMS_STS_PRESENT_CURRENT_H = 70

class sms_sts(protocol_packet_handler):
    def __init__(self, portHandler):
        protocol_packet_handler.__init__(self, portHandler, 0)
        self.groupSyncWrite = GroupSyncWrite(self, SMS_STS_ACC, 7)

    def WritePosEx(self, scs_id, position, speed, acc, time=0):
        """写入位置/速度/加速度/[goal time].

        time: 到达目标位置的毫秒数。0 = 按 GOAL_SPEED 走（默认，向后兼容）;
              非 0 = 强制在指定毫秒内到达（忽略 GOAL_SPEED 作 cap）。
        """
        position = self.scs_toscs(position, 15)
        time = max(0, min(int(time), 65535))
        txpacket = [acc,
                    self.scs_lobyte(position), self.scs_hibyte(position),
                    self.scs_lobyte(time),    self.scs_hibyte(time),
                    self.scs_lobyte(speed),   self.scs_hibyte(speed)]
        return self.writeTxRx(scs_id, SMS_STS_ACC, len(txpacket), txpacket)

    def ReadPos(self, scs_id):
        scs_present_position, scs_comm_result, scs_error = self.read2ByteTxRx(scs_id, SMS_STS_PRESENT_POSITION_L)
        return self.scs_tohost(scs_present_position, 15), scs_comm_result, scs_error

    def ReadSpeed(self, scs_id):
        scs_present_speed, scs_comm_result, scs_error = self.read2ByteTxRx(scs_id, SMS_STS_PRESENT_SPEED_L)
        return self.scs_tohost(scs_present_speed, 15), scs_comm_result, scs_error

    def ReadPosSpeed(self, scs_id):
        scs_present_position_speed, scs_comm_result, scs_error = self.read4ByteTxRx(scs_id, SMS_STS_PRESENT_POSITION_L)
        scs_present_position = self.scs_loword(scs_present_position_speed)
        scs_present_speed = self.scs_hiword(scs_present_position_speed)
        return self.scs_tohost(scs_present_position, 15), self.scs_tohost(scs_present_speed, 15), scs_comm_result, scs_error

    def ReadMoving(self, scs_id):
        moving, scs_comm_result, scs_error = self.read1ByteTxRx(scs_id, SMS_STS_MOVING)
        return moving, scs_comm_result, scs_error

    def SyncWritePosEx(self, scs_id, position, speed, acc, time=0):
        position = self.scs_toscs(position, 15)
        time = max(0, min(int(time), 65535))
        txpacket = [acc,
                    self.scs_lobyte(position), self.scs_hibyte(position),
                    self.scs_lobyte(time),    self.scs_hibyte(time),
                    self.scs_lobyte(speed),   self.scs_hibyte(speed)]
        return self.groupSyncWrite.addParam(scs_id, txpacket)

    def RegWritePosEx(self, scs_id, position, speed, acc, time=0):
        position = self.scs_toscs(position, 15)
        time = max(0, min(int(time), 65535))
        txpacket = [acc,
                    self.scs_lobyte(position), self.scs_hibyte(position),
                    self.scs_lobyte(time),    self.scs_hibyte(time),
                    self.scs_lobyte(speed),   self.scs_hibyte(speed)]
        return self.regWriteTxRx(scs_id, SMS_STS_ACC, len(txpacket), txpacket)

    def RegAction(self):
        return self.action(BROADCAST_ID)

    def WheelMode(self, scs_id):
        return self.write1ByteTxRx(scs_id, SMS_STS_MODE, 1)

    def WriteSpec(self, scs_id, speed, acc):
        speed = self.scs_toscs(speed, 15)
        txpacket = [acc, 0, 0, 0, 0, self.scs_lobyte(speed), self.scs_hibyte(speed)]
        return self.writeTxRx(scs_id, SMS_STS_ACC, len(txpacket), txpacket)

    def LockEprom(self, scs_id):
        return self.write1ByteTxRx(scs_id, SMS_STS_LOCK, 1)

    def unLockEprom(self, scs_id):
        return self.write1ByteTxRx(scs_id, SMS_STS_LOCK, 0)

